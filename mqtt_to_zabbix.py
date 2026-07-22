#!/usr/bin/env python3
"""
mqtt_to_zabbix.py — PredictFlow / Vantive Tunisia

Pont MQTTS (TLS 1.2) → InfluxDB (stockage chaud) + Zabbix (supervision sémantique).

Topics consommés :
  - factory/line1/+/telemetry  → inférence CNN 1D + statistiques de fenêtre
  - esp32/ota/status           → résultats OTA (SUCCESS / FAILED)

Le feedback opérateur ne transite PAS par ce bridge : il est routé directement
en REST depuis le backend Express vers MinIO.

Supervision sémantique : Zabbix reçoit les sorties du modèle embarqué
(tinyml.class, tinyml.confidence), jamais des seuils d'amplitude vibratoire.
Les défauts de roulements étant des phénomènes fréquentiels, aucun seuil
d'amplitude dans le domaine temporel n'est physiquement significatif.
"""

import json
import os
import time
import signal
import sys
import logging
import threading
import warnings
from pathlib import Path

import paho.mqtt.client as mqtt
from pyzabbix import ZabbixMetric, ZabbixSender

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import WriteOptions

# Le .env est résolu à partir de l'emplacement DU SCRIPT, jamais du répertoire
# courant : PM2 démarre les process depuis un cwd arbitraire (ici ~/Bureau),
# ce qui rendrait un load_dotenv() nu silencieusement inopérant.
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ENV_PATH)
except ImportError:
    pass

# L'API de callbacks v1 de paho reste utilisée volontairement (la v2 modifie la
# signature de on_connect). L'avertissement est attribué au frame appelant, donc
# le filtre porte sur le message et non sur le module paho.
warnings.filterwarnings(
    "ignore",
    message=r".*Callback API version 1 is deprecated.*",
    category=DeprecationWarning,
)

# --------------------
# Logging
# --------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# --------------------
# Config
# --------------------
ZABBIX_SERVER   = os.getenv("ZABBIX_SERVER", "127.0.0.1")
ZABBIX_PORT     = int(os.getenv("ZABBIX_PORT", "10051"))
ZABBIX_HOSTNAME = os.getenv("ZABBIX_HOSTNAME", "ESP32_Host")

MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.137.100")
MQTT_PORT   = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USER   = os.getenv("MQTT_USER", "esp32_s3")
MQTT_PASS   = os.getenv("MQTT_PASS", "esp32_pass")
CA_CERT     = os.getenv("CA_CERT", str(BASE_DIR / "certs" / "ca.crt"))

MQTT_TOPIC_TELEMETRY = "factory/line1/+/telemetry"
MQTT_TOPIC_OTA       = "esp32/ota/status"

INFLUX_URL    = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN  = os.getenv("INFLUX_TOKEN")
INFLUX_ORG    = os.getenv("INFLUX_ORG", "stage_pfe")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "esp32-sensor")

MEASUREMENT     = "esp32_telemetry"
MEASUREMENT_OTA = "ota_events"

REQUIRED_KEYS = {"device"}

# 4 classes CWRU — ordre alphabétique LabelEncoder sklearn
VALID_CLASSES = {"ball", "inner_race", "normal", "outer_race"}

# Heartbeat envoyé à Zabbix pour le trigger « perte du script d'ingestion »
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "30"))

CLOCK_DRIFT_TOLERANCE = 5      # secondes
EPOCH_SANITY_FLOOR    = 1_700_000_000   # ~nov. 2023 : filtre les uptime bruts


# --------------------
# InfluxDB batch callbacks
# --------------------
def influx_success_cb(conf, data):
    log.debug(f"InfluxDB batch written: {conf}")

def influx_error_cb(conf, data, exception):
    log.error(f"InfluxDB batch FAILED: {exception}")

def influx_retry_cb(conf, data, exception):
    log.warning(f"InfluxDB batch retry: {exception}")


# --------------------
# TelemetryBridge
# --------------------
class TelemetryBridge:

    def __init__(self):
        if not INFLUX_TOKEN:
            log.error(f"INFLUX_TOKEN absent. Fichier .env attendu : {ENV_PATH}")
            log.error(f"  (présent sur le disque : {ENV_PATH.exists()})")
            sys.exit(1)

        if not Path(CA_CERT).exists():
            log.warning(f"Certificat CA introuvable : {CA_CERT}")

        # --- InfluxDB (batch asynchrone) ---
        self.influx_client = InfluxDBClient(
            url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG
        )
        self.write_api = self.influx_client.write_api(
            write_options=WriteOptions(
                batch_size=50,
                flush_interval=1_000,
                jitter_interval=200,
                retry_interval=5_000,
                max_retries=3,
                max_retry_delay=30_000,
                exponential_base=2,
            ),
            success_callback=influx_success_cb,
            error_callback=influx_error_cb,
            retry_callback=influx_retry_cb,
        )

        # --- Zabbix ---
        self.zabbix_sender = ZabbixSender(ZABBIX_SERVER, ZABBIX_PORT)

        # --- MQTT (MQTTS TLS 1.2) ---
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        self.mqtt_client.tls_set(ca_certs=CA_CERT)
        self.mqtt_client.tls_insecure_set(True)   # dette technique — fix post-soutenance
        self.mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        self.mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)

        # --- Heartbeat ---
        self._stop_event = threading.Event()
        self._last_device = None
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )

        log.info("TelemetryBridge initialisé ✅")

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def safe_get(dct, path, default=None):
        cur = dct
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    @staticmethod
    def validate_payload(data: dict) -> bool:
        return REQUIRED_KEYS.issubset(data.keys())

    @staticmethod
    def as_int(value, default=None):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def as_float(value, default=None, digits=None):
        try:
            v = float(value)
            return round(v, digits) if digits is not None else v
        except (TypeError, ValueError):
            return default

    # ── Heartbeat ────────────────────────────────────────────

    def _heartbeat_loop(self):
        """Signale la vitalité du bridge à Zabbix (trigger : absence > 60s)."""
        while not self._stop_event.is_set():
            host = f"ESP32_{self._last_device}" if self._last_device else ZABBIX_HOSTNAME
            try:
                self.zabbix_sender.send([ZabbixMetric(host, "bridge.heartbeat", "1")])
                log.debug(f"Heartbeat envoyé à Zabbix ({host})")
            except Exception as e:
                log.debug(f"Heartbeat non délivré: {e}")
            self._stop_event.wait(HEARTBEAT_INTERVAL)

    # ── MQTT callbacks ───────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            log.info(f"MQTTS connecté sur {MQTT_BROKER}:{MQTT_PORT} (TLS 1.2) ✅")
            client.subscribe(MQTT_TOPIC_TELEMETRY)
            client.subscribe(MQTT_TOPIC_OTA)
            log.info(f"Souscrit → {MQTT_TOPIC_TELEMETRY}")
            log.info(f"Souscrit → {MQTT_TOPIC_OTA}")
        else:
            rc_messages = {
                1: "Version protocole refusée",
                2: "Client ID refusé",
                3: "Broker indisponible",
                4: "Mauvais identifiants (MQTT_USER/PASS)",
                5: "Non autorisé (ACL)"
            }
            log.error(f"MQTT connexion échouée rc={rc} — {rc_messages.get(rc, 'Erreur inconnue')}")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = msg.payload.decode("utf-8", errors="replace")
            data    = json.loads(payload)

            if topic == MQTT_TOPIC_OTA:
                self._handle_ota(data)
            else:
                self._handle_telemetry(topic, data)

        except json.JSONDecodeError as e:
            log.error(f"JSON decode error on {topic}: {e}")
        except Exception as e:
            log.error(f"Erreur traitement message {topic}: {e}", exc_info=True)

    # ── Telemetry handler ────────────────────────────────────

    def _handle_telemetry(self, topic, data):
        if not self.validate_payload(data):
            log.warning(f"Payload invalide sur {topic}: {str(data)[:200]}")
            return

        device    = data.get("device", "unknown")
        esp32_ts  = data.get("timestamp")
        msg_id    = data.get("msg_id")
        server_ts = int(time.time())
        self._last_device = device

        # Détection de dérive d'horloge (l'ESP32-S3 se synchronise via NTP ;
        # un uptime brut est filtré par le plancher de vraisemblance).
        if isinstance(esp32_ts, (int, float)) and esp32_ts > EPOCH_SANITY_FLOOR:
            drift = abs(server_ts - int(esp32_ts))
            if drift > CLOCK_DRIFT_TOLERANCE:
                log.warning(f"Dérive horloge ESP32: {drift}s (device={device})")
        elif esp32_ts is not None:
            log.debug(f"Horodatage non synchronisé (uptime brut) — device={device}")

        # Statistiques de fenêtre. CWRU est un signal MONO-AXE (accéléromètre
        # drive-end, clé DE_time) : on remonte RMS / crête / écart-type et non
        # trois axes qui laisseraient croire à un capteur triaxial.
        v_rms     = self.as_float(self.safe_get(data, ("sensors", "vibration_rms")),  digits=4)
        v_peak    = self.as_float(self.safe_get(data, ("sensors", "vibration_peak")), digits=4)
        v_std     = self.as_float(self.safe_get(data, ("sensors", "vibration_std")),  digits=4)
        wifi_rssi = self.as_int(self.safe_get(data, ("sensors", "wifi_rssi")))

        # TinyML CNN 1D — 4 classes CWRU
        prediction    = self.safe_get(data, ("tinyml", "prediction"))
        confidence    = self.as_float(self.safe_get(data, ("tinyml", "confidence")), digits=3)
        fault_class   = self.safe_get(data, ("tinyml", "fault_class"))
        anomaly       = self.safe_get(data, ("tinyml", "anomaly"))
        model_version = self.safe_get(data, ("tinyml", "version"))

        # Performance embarquée → alimente le tableau comparatif du rapport
        cnn_ms   = self.as_int(self.safe_get(data, ("performance", "cnn1d_ms")))
        arena_kb = self.as_int(self.safe_get(data, ("performance", "arena_used_kb")))
        ram_kb   = self.as_int(self.safe_get(data, ("performance", "ram_free_kb")))
        psram_kb = self.as_int(self.safe_get(data, ("performance", "psram_free_kb")))

        # Benchmark MLP — présent uniquement avec le firmware compilé ENABLE_MLP=1
        mlp_class = self.safe_get(data, ("benchmark", "mlp_class"))
        mlp_conf  = self.as_float(self.safe_get(data, ("benchmark", "mlp_confidence")), digits=3)
        mlp_ms    = self.as_int(self.safe_get(data, ("benchmark", "mlp_ms")))

        # Vérité terrain → accuracy et matrice de confusion réelles on-device
        sim_class = data.get("sim_class")
        correct   = data.get("correct")

        if fault_class and fault_class not in VALID_CLASSES:
            log.warning(f"Classe inconnue reçue: '{fault_class}' — attendu: {VALID_CLASSES}")

        verdict = "" if correct is None else (" ✓" if correct else " ✗")
        log.info(f"TELEMETRY device={device} msg_id={msg_id} "
                 f"reel={sim_class} predit={fault_class} conf={confidence} "
                 f"latence={cnn_ms}ms{verdict}")

        self._send_to_zabbix(device, v_rms, v_peak, wifi_rssi,
                             prediction, confidence, fault_class, anomaly, cnn_ms)

        self._write_to_influx(
            device=device, server_ts=server_ts, esp32_ts=esp32_ts, msg_id=msg_id,
            v_rms=v_rms, v_peak=v_peak, v_std=v_std, wifi_rssi=wifi_rssi,
            prediction=prediction, confidence=confidence, fault_class=fault_class,
            anomaly=anomaly, model_version=model_version,
            cnn_ms=cnn_ms, arena_kb=arena_kb, ram_kb=ram_kb, psram_kb=psram_kb,
            mlp_class=mlp_class, mlp_conf=mlp_conf, mlp_ms=mlp_ms,
            sim_class=sim_class, correct=correct,
        )

    # ── OTA handler ──────────────────────────────────────────

    def _handle_ota(self, data):
        device  = data.get("device", "unknown")
        status  = data.get("status", "UNKNOWN")
        version = data.get("version", "unknown")

        log.info(f"OTA event: device={device} status={status} version={version}")

        try:
            p = (
                Point(MEASUREMENT_OTA)
                .tag("device", device)
                .tag("status", status)
                .field("version", str(version))
                .field("raw", json.dumps(data))
                .time(int(time.time()), WritePrecision.S)
            )
            self.write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
            log.debug("OTA event écrit dans InfluxDB ✅")
        except Exception as e:
            log.warning(f"InfluxDB OTA write failed: {e}")

        zabbix_host = f"ESP32_{device}" if device != "unknown" else ZABBIX_HOSTNAME
        try:
            self.zabbix_sender.send([ZabbixMetric(zabbix_host, "ota.status", str(status))])
            if status == "FAILED":
                log.warning(f"Alerte Zabbix OTA FAILED envoyée pour {device}")
        except Exception as e:
            log.warning(f"Zabbix OTA send failed: {e}")

    # ── Zabbix ───────────────────────────────────────────────

    def _send_to_zabbix(self, device, v_rms, v_peak, wifi_rssi,
                        prediction, confidence, fault_class, anomaly, cnn_ms):
        """
        Les métriques vibratoires sont remontées à titre informatif uniquement.
        Aucun trigger n'est défini dessus : l'alerting repose exclusivement sur
        les sorties sémantiques du modèle (tinyml.class, tinyml.confidence).
        """
        zabbix_host = f"ESP32_{device}" if device != "unknown" else ZABBIX_HOSTNAME
        metrics = []

        if v_rms      is not None: metrics.append(ZabbixMetric(zabbix_host, "vibration_rms",      str(v_rms)))
        if v_peak     is not None: metrics.append(ZabbixMetric(zabbix_host, "vibration_peak",     str(v_peak)))
        if wifi_rssi  is not None: metrics.append(ZabbixMetric(zabbix_host, "wifi_rssi",          str(wifi_rssi)))
        if cnn_ms     is not None: metrics.append(ZabbixMetric(zabbix_host, "tinyml.latency",     str(cnn_ms)))
        if prediction is not None: metrics.append(ZabbixMetric(zabbix_host, "tinyml.prediction",  str(prediction)))
        if confidence is not None: metrics.append(ZabbixMetric(zabbix_host, "tinyml.confidence",  str(confidence)))
        if fault_class is not None: metrics.append(ZabbixMetric(zabbix_host, "tinyml.class",      str(fault_class)))
        if anomaly    is not None: metrics.append(ZabbixMetric(zabbix_host, "tinyml.anomaly",     str(int(bool(anomaly)))))

        if metrics:
            try:
                result = self.zabbix_sender.send(metrics)
                failed = getattr(result, "failed", 0)
                if failed:
                    log.warning(
                        f"Zabbix: {failed}/{getattr(result, 'total', len(metrics))} métriques rejetées — "
                        f"l'host « {zabbix_host} » ou ses items trappeurs n'existent pas côté serveur."
                    )
                else:
                    log.info(f"Zabbix: {result}")
            except Exception as e:
                log.warning(f"Zabbix send failed: {e}")
        else:
            log.warning("Zabbix: aucune métrique à envoyer")

    # ── InfluxDB ─────────────────────────────────────────────

    def _write_to_influx(self, device, server_ts, esp32_ts, msg_id,
                         v_rms, v_peak, v_std, wifi_rssi,
                         prediction, confidence, fault_class,
                         anomaly, model_version,
                         cnn_ms, arena_kb, ram_kb, psram_kb,
                         mlp_class, mlp_conf, mlp_ms,
                         sim_class, correct):
        try:
            p = Point(MEASUREMENT).tag("device", device)

            # Signal vibratoire (statistiques de fenêtre, mono-axe)
            if v_rms     is not None: p = p.field("vibration_rms",  float(v_rms))
            if v_peak    is not None: p = p.field("vibration_peak", float(v_peak))
            if v_std     is not None: p = p.field("vibration_std",  float(v_std))
            if wifi_rssi is not None: p = p.field("wifi_rssi",      int(wifi_rssi))

            # Sorties du modèle embarqué
            if prediction    is not None: p = p.field("prediction",    str(prediction))
            if confidence    is not None: p = p.field("confidence",    float(confidence))
            if fault_class   is not None: p = p.field("class",         str(fault_class))
            if anomaly       is not None: p = p.field("anomaly",       int(bool(anomaly)))
            if model_version is not None: p = p.field("model_version", str(model_version))

            # Performance embarquée
            if cnn_ms   is not None: p = p.field("cnn1d_ms",      int(cnn_ms))
            if arena_kb is not None: p = p.field("arena_used_kb", int(arena_kb))
            if ram_kb   is not None: p = p.field("ram_free_kb",   int(ram_kb))
            if psram_kb is not None: p = p.field("psram_free_kb", int(psram_kb))

            # Benchmark comparatif MLP (firmware ENABLE_MLP=1 uniquement)
            if mlp_class is not None: p = p.field("mlp_class",      str(mlp_class))
            if mlp_conf  is not None: p = p.field("mlp_confidence", float(mlp_conf))
            if mlp_ms    is not None: p = p.field("mlp_ms",         int(mlp_ms))

            # Vérité terrain
            if sim_class is not None: p = p.field("sim_class", str(sim_class))
            if correct   is not None: p = p.field("correct",   int(bool(correct)))

            if msg_id is not None:
                try:    p = p.field("msg_id", int(msg_id))
                except: p = p.field("msg_id", str(msg_id))

            if isinstance(esp32_ts, (int, float)) and esp32_ts > EPOCH_SANITY_FLOOR:
                p = p.field("esp32_ts", int(esp32_ts))

            p = p.time(server_ts, WritePrecision.S)

            self.write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
            log.debug("InfluxDB: point en queue ✅")

        except Exception as e:
            log.warning(f"InfluxDB write failed: {e}")

    # ── Lifecycle ────────────────────────────────────────────

    def start(self):
        log.info(f"Connexion au broker MQTTS {MQTT_BROKER}:{MQTT_PORT}...")
        self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
        self._heartbeat_thread.start()
        log.info(f"Heartbeat Zabbix actif (toutes les {HEARTBEAT_INTERVAL}s)")
        self.mqtt_client.loop_forever()

    def close(self):
        log.info("Arrêt TelemetryBridge...")
        self._stop_event.set()
        try:
            self.write_api.close()
            self.influx_client.close()
            self.mqtt_client.disconnect()
            log.info("Tous les clients fermés ✅")
        except Exception as e:
            log.error(f"Erreur lors de l'arrêt: {e}")


# --------------------
# Main
# --------------------
def main():
    bridge = TelemetryBridge()

    def handle_signal(sig, frame):
        log.info(f"Signal {sig} reçu, arrêt en cours...")
        bridge.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    bridge.start()


if __name__ == "__main__":
    main()
