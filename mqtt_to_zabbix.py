import json
import time
import signal
import sys
import logging
import paho.mqtt.client as mqtt
from pyzabbix import ZabbixMetric, ZabbixSender

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import WriteOptions

# --------------------
# Logging (replaces print() for production-grade output)
# --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# --------------------
# Config
# --------------------
ZABBIX_SERVER = "127.0.0.1"
ZABBIX_PORT = 10051
ZABBIX_HOSTNAME = "ESP32_Host"   # doit matcher EXACTEMENT le hostname dans Zabbix

MQTT_BROKER = "127.0.0.1"        # si EMQX est sur la même VM
MQTT_PORT = 1883
MQTT_TOPIC = "factory/line1/+/telemetry"  # wildcard: tous les devices de line1

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "K78LjzMs2yCJP6tgdxVqFf0IV4P8rxA82zPb1gcCgwx7FngDgub7DrjjrL3SnNgDqKFdW_JmleIySWX6qwCssQ=="
INFLUX_ORG = "stage_pfe"
INFLUX_BUCKET = "esp32-sensor"

MEASUREMENT = "esp32_telemetry"

# Clés obligatoires dans le payload JSON
REQUIRED_KEYS = {"device", "sensors"}

# --------------------
# Callbacks d'erreur pour le mode batch InfluxDB
# --------------------
def influx_success_cb(conf, data):
    log.debug(f"InfluxDB batch written: {conf}")

def influx_error_cb(conf, data, exception):
    log.error(f"InfluxDB batch FAILED: {exception}")

def influx_retry_cb(conf, data, exception):
    log.warning(f"InfluxDB batch retry: {exception}")


# --------------------
# TelemetryBridge : encapsule tous les clients
# --------------------
class TelemetryBridge:
    """Pont MQTT → Zabbix + InfluxDB.
    
    Tous les clients (InfluxDB, Zabbix, MQTT) sont initialisés une seule
    fois et réutilisés pour chaque message. Le mode batch asynchrone
    d'InfluxDB évite de bloquer la boucle MQTT.
    """

    def __init__(self):
        # --- InfluxDB (batch asynchrone) ---
        self.influx_client = InfluxDBClient(
            url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG
        )
        self.write_api = self.influx_client.write_api(
            write_options=WriteOptions(
                batch_size=50,           # écrit par paquets de 50 points
                flush_interval=1_000,    # flush toutes les 1s max
                jitter_interval=200,     # jitter pour éviter les pics
                retry_interval=5_000,    # réessai après 5s en cas d'erreur
                max_retries=3,
                max_retry_delay=30_000,
                exponential_base=2,
            ),
            success_callback=influx_success_cb,
            error_callback=influx_error_cb,
            retry_callback=influx_retry_cb,
        )

        # --- Zabbix (réutilisé, pas de reconnexion par message) ---
        self.zabbix_sender = ZabbixSender(ZABBIX_SERVER, ZABBIX_PORT)

        # --- MQTT ---
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        self.mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)

        log.info("TelemetryBridge initialized (InfluxDB=batch_async, Zabbix=reused)")

    # ---- helpers ----

    @staticmethod
    def safe_get(dct, path, default=None):
        """Accès imbriqué sécurisé. path ex: ('sensors','temperature')"""
        cur = dct
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    @staticmethod
    def validate_payload(data: dict) -> bool:
        """Vérifie que le payload contient les clés minimales requises."""
        if not REQUIRED_KEYS.issubset(data.keys()):
            return False
        if not isinstance(data.get("sensors"), dict):
            return False
        return True

    # ---- MQTT callbacks ----

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            log.info(f"MQTT connected rc={rc}")
            client.subscribe(MQTT_TOPIC)
            log.info(f"Subscribed to {MQTT_TOPIC}")
        else:
            log.error(f"MQTT connection failed rc={rc}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8", errors="replace")
            data = json.loads(payload)

            # --- Validation schéma ---
            if not self.validate_payload(data):
                log.warning(f"Invalid payload schema on {msg.topic}: {payload[:200]}")
                return

            device = data.get("device", "unknown")
            esp32_ts = data.get("timestamp")  # epoch seconds (from ESP32)
            msg_id = data.get("msg_id")

            # Timestamp serveur Python (fiable, synchronisé)
            server_ts = int(time.time())

            # Détection de drift d'horloge ESP32
            if isinstance(esp32_ts, (int, float)) and esp32_ts > 100000:
                drift = abs(server_ts - int(esp32_ts))
                if drift > 5:
                    log.warning(f"Clock drift ESP32: {drift}s (device={device})")

            # --- Sensors ---
            temperature = self.safe_get(data, ("sensors", "temperature"))
            humidity = self.safe_get(data, ("sensors", "humidity"))
            vx = self.safe_get(data, ("sensors", "vibration_x"))
            vy = self.safe_get(data, ("sensors", "vibration_y"))
            vz = self.safe_get(data, ("sensors", "vibration_z"))
            wifi_rssi = self.safe_get(data, ("sensors", "wifi_rssi"))

            # --- TinyML ---
            prediction = self.safe_get(data, ("tinyml", "prediction"))
            confidence = self.safe_get(data, ("tinyml", "confidence"))
            fault_class = self.safe_get(data, ("tinyml", "class"))

            log.info(f"MSG topic={msg.topic} device={device} msg_id={msg_id}")

            # --------------------
            # Send to Zabbix (isolated error handling)
            # --------------------
            self._send_to_zabbix(
                device, temperature, humidity, vx, vy, vz,
                wifi_rssi, prediction, confidence, fault_class
            )

            # --------------------
            # Write to InfluxDB (batch async — non-bloquant)
            # --------------------
            self._write_to_influx(
                device, server_ts, esp32_ts, msg_id,
                temperature, humidity, vx, vy, vz,
                wifi_rssi, prediction, confidence, fault_class
            )

        except json.JSONDecodeError as e:
            log.error(f"JSON decode error on {msg.topic}: {e}")
        except Exception as e:
            log.error(f"Error processing message: {e}", exc_info=True)

    # ---- Zabbix ----

    def _send_to_zabbix(self, device, temperature, humidity, vx, vy, vz,
                        wifi_rssi, prediction, confidence, fault_class):
        # Dynamic hostname: use device name so multiple ESP32s don't collide
        zabbix_host = f"ESP32_{device}" if device != "unknown" else ZABBIX_HOSTNAME

        metrics = []

        # N'envoie que si la valeur existe (évite KeyError)
        if temperature is not None:
            metrics.append(ZabbixMetric(zabbix_host, "temperature", str(temperature)))
        if humidity is not None:
            metrics.append(ZabbixMetric(zabbix_host, "humidity", str(humidity)))
        if vx is not None:
            metrics.append(ZabbixMetric(zabbix_host, "vibration_x", str(vx)))
        if vy is not None:
            metrics.append(ZabbixMetric(zabbix_host, "vibration_y", str(vy)))
        if vz is not None:
            metrics.append(ZabbixMetric(zabbix_host, "vibration_z", str(vz)))
        if wifi_rssi is not None:
            metrics.append(ZabbixMetric(zabbix_host, "wifi_rssi", str(wifi_rssi)))

        if prediction is not None:
            metrics.append(ZabbixMetric(zabbix_host, "tinyml.prediction", str(prediction)))
        if confidence is not None:
            metrics.append(ZabbixMetric(zabbix_host, "tinyml.confidence", str(confidence)))
        if fault_class is not None:
            metrics.append(ZabbixMetric(zabbix_host, "tinyml.class", str(fault_class)))

        if metrics:
            try:
                result = self.zabbix_sender.send(metrics)
                log.info(f"Zabbix: {result}")
            except Exception as e:
                log.warning(f"Zabbix send failed: {e}")
        else:
            log.warning("Zabbix: no metrics to send (empty payload?)")

    # ---- InfluxDB ----

    def _write_to_influx(self, device, server_ts, esp32_ts, msg_id,
                         temperature, humidity, vx, vy, vz,
                         wifi_rssi, prediction, confidence, fault_class):
        try:
            p = Point(MEASUREMENT).tag("device", device)

            # Ajoute fields si présents
            if temperature is not None:
                p = p.field("temperature", float(temperature))
            if humidity is not None:
                p = p.field("humidity", float(humidity))
            if vx is not None:
                p = p.field("vibration_x", float(vx))
            if vy is not None:
                p = p.field("vibration_y", float(vy))
            if vz is not None:
                p = p.field("vibration_z", float(vz))
            if wifi_rssi is not None:
                p = p.field("wifi_rssi", int(wifi_rssi))

            # TinyML
            if prediction is not None:
                p = p.field("prediction", str(prediction))
            if confidence is not None:
                p = p.field("confidence", float(confidence))
            if fault_class is not None:
                p = p.field("class", str(fault_class))

            if msg_id is not None:
                try:
                    p = p.field("msg_id", int(msg_id))
                except Exception:
                    p = p.field("msg_id", str(msg_id))

            # Conserve le timestamp ESP32 comme field de diagnostic
            if isinstance(esp32_ts, (int, float)) and esp32_ts > 100000:
                p = p.field("esp32_ts", int(esp32_ts))

            # Timestamp serveur Python (fiable, synchronisé NTP)
            p = p.time(server_ts, WritePrecision.S)

            # Écriture non-bloquante (batch async)
            self.write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
            log.debug("InfluxDB: point queued ✅")

        except Exception as e:
            log.warning(f"InfluxDB write failed: {e}")

    # ---- lifecycle ----

    def start(self):
        """Démarre la boucle MQTT (bloquante)."""
        log.info(f"Connecting to MQTT broker {MQTT_BROKER}:{MQTT_PORT}")
        self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
        self.mqtt_client.loop_forever()

    def close(self):
        """Ferme proprement tous les clients."""
        log.info("Shutting down TelemetryBridge...")
        try:
            self.write_api.close()       # flush les points en attente
            self.influx_client.close()
            self.mqtt_client.disconnect()
            log.info("All clients closed ✅")
        except Exception as e:
            log.error(f"Error during shutdown: {e}")


# --------------------
# Main
# --------------------
def main():
    bridge = TelemetryBridge()

    # Gestion propre de SIGTERM/SIGINT (systemd envoie SIGTERM)
    def handle_signal(sig, frame):
        log.info(f"Signal {sig} received, shutting down...")
        bridge.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    bridge.start()


if __name__ == "__main__":
    main()
