import json
import os
from dotenv import load_dotenv
load_dotenv()
import time
import signal
import sys
import logging
import paho.mqtt.client as mqtt
from pyzabbix import ZabbixMetric, ZabbixSender

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import WriteOptions

# --------------------
# Logging
# --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# --------------------
# Config
# --------------------
ZABBIX_SERVER   = "127.0.0.1"
ZABBIX_PORT     = 10051
ZABBIX_HOSTNAME = "ESP32_Host"

MQTT_BROKER = "192.168.137.100"
MQTT_PORT   = 8883
MQTT_USER   = "esp32_s3"
MQTT_PASS   = "esp32_pass"
CA_CERT     = "/home/yassine/PFE_IOT/certs/ca.crt"

MQTT_TOPIC_TELEMETRY = "factory/line1/+/telemetry"
MQTT_TOPIC_OTA       = "esp32/ota/status"

INFLUX_URL    = "http://localhost:8086"
INFLUX_TOKEN  = os.getenv("INFLUX_TOKEN")
INFLUX_ORG    = "stage_pfe"
INFLUX_BUCKET = "esp32-sensor"

MEASUREMENT     = "esp32_telemetry"
MEASUREMENT_OTA = "ota_events"

REQUIRED_KEYS = {"device"}

# 4 classes CWRU — entraînement local Keras/TFLM
VALID_CLASSES = {"ball", "inner_race", "normal", "outer_race"}

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
    """
    Pont MQTTS (TLS 1.2) → Zabbix + InfluxDB.
    Topics gérés :
      - factory/line1/+/telemetry  → inférence TinyML CNN 1D + vibrations
      - esp32/ota/status           → résultats OTA (SUCCESS / FAILED)
    Note : feedback opérateur supprimé — routé directement REST Express → MinIO.
    """

    def __init__(self):
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

    # ── MQTT callbacks ────────────────────────────────────────

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

    # ── Telemetry handler ─────────────────────────────────────

    def _handle_telemetry(self, topic, data):
        if not self.validate_payload(data):
            log.warning(f"Payload invalide sur {topic}: {str(data)[:200]}")
            return

        device    = data.get("device", "unknown")
        esp32_ts  = data.get("timestamp")
        msg_id    = data.get("msg_id")
        server_ts = int(time.time())

        # Détection dérive horloge ESP32
        if isinstance(esp32_ts, (int, float)) and esp32_ts > 100000:
            drift = abs(server_ts - int(esp32_ts))
            if drift > 5:
                log.warning(f"Dérive horloge ESP32: {drift}s (device={device})")

        # Vibrations uniquement — pas de température/humidité (pas de capteur physique)
        vx        = self.safe_get(data, ("sensors", "vibration_x"))
        vy        = self.safe_get(data, ("sensors", "vibration_y"))
        vz        = self.safe_get(data, ("sensors", "vibration_z"))
        wifi_rssi = self.safe_get(data, ("sensors", "wifi_rssi"))

        # TinyML CNN 1D — 4 classes CWRU
        prediction    = self.safe_get(data, ("tinyml", "prediction"))
        confidence    = self.safe_get(data, ("tinyml", "confidence"))
        fault_class   = self.safe_get(data, ("tinyml", "fault_class"))
        anomaly       = self.safe_get(data, ("tinyml", "anomaly"))
        model_version = self.safe_get(data, ("tinyml", "version"))

        if fault_class and fault_class not in VALID_CLASSES:
            log.warning(f"Classe inconnue reçue: '{fault_class}' — attendu: {VALID_CLASSES}")

        log.info(f"TELEMETRY device={device} msg_id={msg_id} "
                 f"class={fault_class} conf={confidence} anomaly={anomaly}")

        self._send_to_zabbix(device, vx, vy, vz, wifi_rssi,
                             prediction, confidence, fault_class, anomaly)
        self._write_to_influx(device, server_ts, esp32_ts, msg_id,
                              vx, vy, vz, wifi_rssi,
                              prediction, confidence, fault_class,
                              anomaly, model_version)

    # ── OTA handler ───────────────────────────────────────────

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

        if status == "FAILED":
            try:
                zabbix_host = f"ESP32_{device}"
                self.zabbix_sender.send([ZabbixMetric(zabbix_host, "ota.status", "FAILED")])
                log.warning(f"Alerte Zabbix OTA FAILED envoyée pour {device}")
            except Exception as e:
                log.warning(f"Zabbix OTA alert failed: {e}")

    # ── Zabbix ───────────────────────────────────────────────

    def _send_to_zabbix(self, device, vx, vy, vz, wifi_rssi,
                        prediction, confidence, fault_class, anomaly):
        zabbix_host = f"ESP32_{device}" if device != "unknown" else ZABBIX_HOSTNAME
        metrics = []

        if vx          is not None: metrics.append(ZabbixMetric(zabbix_host, "vibration_x",        str(vx)))
        if vy          is not None: metrics.append(ZabbixMetric(zabbix_host, "vibration_y",        str(vy)))
        if vz          is not None: metrics.append(ZabbixMetric(zabbix_host, "vibration_z",        str(vz)))
        if wifi_rssi   is not None: metrics.append(ZabbixMetric(zabbix_host, "wifi_rssi",          str(wifi_rssi)))
        if prediction  is not None: metrics.append(ZabbixMetric(zabbix_host, "tinyml.prediction",  str(prediction)))
        if confidence  is not None: metrics.append(ZabbixMetric(zabbix_host, "tinyml.confidence",  str(confidence)))
        if fault_class is not None: metrics.append(ZabbixMetric(zabbix_host, "tinyml.class",       str(fault_class)))
        if anomaly     is not None: metrics.append(ZabbixMetric(zabbix_host, "tinyml.anomaly",     str(int(anomaly))))

        if metrics:
            try:
                result = self.zabbix_sender.send(metrics)
                log.info(f"Zabbix: {result}")
            except Exception as e:
                log.warning(f"Zabbix send failed: {e}")
        else:
            log.warning("Zabbix: aucune métrique à envoyer")

    # ── InfluxDB ─────────────────────────────────────────────

    def _write_to_influx(self, device, server_ts, esp32_ts, msg_id,
                         vx, vy, vz, wifi_rssi,
                         prediction, confidence, fault_class,
                         anomaly, model_version):
        try:
            p = Point(MEASUREMENT).tag("device", device)

            if vx            is not None: p = p.field("vibration_x",   float(vx))
            if vy            is not None: p = p.field("vibration_y",   float(vy))
            if vz            is not None: p = p.field("vibration_z",   float(vz))
            if wifi_rssi     is not None: p = p.field("wifi_rssi",     int(wifi_rssi))
            if prediction    is not None: p = p.field("prediction",    str(prediction))
            if confidence    is not None: p = p.field("confidence",    float(confidence))
            if fault_class   is not None: p = p.field("class",         str(fault_class))
            if anomaly       is not None: p = p.field("anomaly",       int(anomaly))
            if model_version is not None: p = p.field("model_version", str(model_version))

            if msg_id is not None:
                try:    p = p.field("msg_id", int(msg_id))
                except: p = p.field("msg_id", str(msg_id))

            if isinstance(esp32_ts, (int, float)) and esp32_ts > 100000:
                p = p.field("esp32_ts", int(esp32_ts))

            p = p.time(server_ts, WritePrecision.S)

            self.write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
            log.debug("InfluxDB: point en queue ✅")

        except Exception as e:
            log.warning(f"InfluxDB write failed: {e}")

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self):
        log.info(f"Connexion au broker MQTTS {MQTT_BROKER}:{MQTT_PORT}...")
        self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
        self.mqtt_client.loop_forever()

    def close(self):
        log.info("Arrêt TelemetryBridge...")
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
