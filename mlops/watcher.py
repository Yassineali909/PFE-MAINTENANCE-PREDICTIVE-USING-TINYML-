"""
watcher.py — MLflow Production Watcher
=======================================
Monitors MLflow Model Registry every 30s.
When a model is promoted to "Production":
  1. Downloads the .pkl artifact
  2. Creates a firmware package in ota-server/firmware/
  3. Updates version.json so ESP32 knows a new version is available
  4. Logs the OTA event to InfluxDB

Run: python3 watcher.py
"""

import os, json, time, logging, shutil, hashlib
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
import influxdb_client
from influxdb_client import Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --------------------
# Config
# --------------------
MLFLOW_TRACKING  = os.getenv("MLFLOW_TRACKING",  "http://localhost:5000")
INFLUXDB_URL     = os.getenv("INFLUXDB_URL",      "http://localhost:8086")
INFLUXDB_TOKEN   = os.getenv("INFLUXDB_TOKEN",    "K78LjzMs2yCJP6tgdxVqFf0IV4P8rxA82zPb1gcCgwx7FngDgub7DrjjrL3SnNgDqKFdW_JmleIySWX6qwCssQ==")
INFLUXDB_ORG     = os.getenv("INFLUXDB_ORG",      "stage_pfe")
INFLUXDB_BUCKET  = os.getenv("INFLUXDB_BUCKET",   "esp32-sensor")

# Path to OTA server firmware folder (relative to PFE_IOT root)
OTA_FIRMWARE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "ota-server", "firmware"
)
OTA_FIRMWARE_DIR = os.path.abspath(OTA_FIRMWARE_DIR)

POLL_INTERVAL_S  = 30   # check every 30 seconds
STATE_FILE       = os.path.join(os.path.dirname(__file__), ".watcher_state.json")

# --------------------
# State management
# Tracks which model versions have already been processed
# --------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# --------------------
# InfluxDB — log OTA events
# --------------------
def log_ota_event(model_name, version, status, details=""):
    try:
        client = influxdb_client.InfluxDBClient(
            url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG
        )
        write_api = client.write_api(write_options=SYNCHRONOUS)
        p = (
            Point("ota_events")
            .tag("model_name", model_name)
            .tag("status", status)
            .field("version", version)
            .field("details", details)
            .time(int(time.time()), WritePrecision.S)
        )
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=p)
        client.close()
        log.info(f"  OTA event logged to InfluxDB: {model_name} v{version} → {status}")
    except Exception as e:
        log.warning(f"  InfluxDB OTA log failed: {e}")

# --------------------
# OTA firmware deployment
# --------------------
def deploy_to_ota(model_name, version, run_id, client):
    """
    Downloads model artifact from MLflow and creates OTA firmware package.
    For sklearn models: packages the .pkl as a .bin (simulated firmware).
    For CNN1D: uses the Edge Impulse compiled binary if available.
    """
    log.info(f"  Deploying {model_name} v{version} to OTA server...")
    os.makedirs(OTA_FIRMWARE_DIR, exist_ok=True)

    try:
        # Download artifacts from MLflow
        local_dir = f"/tmp/mlflow_artifacts_{run_id}"
        client.download_artifacts(run_id, "model_pkl", local_dir)

        # Find the .pkl file
        pkl_files = []
        for root, dirs, files in os.walk(local_dir):
            for f in files:
                if f.endswith(".pkl"):
                    pkl_files.append(os.path.join(root, f))

        if not pkl_files:
            log.warning(f"  No .pkl found in artifacts for run {run_id}")
            return False

        pkl_src = pkl_files[0]

        # Create firmware filename with version
        safe_name    = model_name.lower().replace(" ", "_")
        firmware_ver = f"{safe_name}_v{version}"
        bin_dst      = os.path.join(OTA_FIRMWARE_DIR, f"firmware_{firmware_ver}.bin")

        # Copy .pkl as .bin (in real ESP32 OTA this would be the compiled .bin)
        shutil.copy2(pkl_src, bin_dst)

        # Compute SHA256 checksum
        with open(bin_dst, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        file_size = os.path.getsize(bin_dst)

        # Update version.json
        version_info = {
            "version":      firmware_ver,
            "model":        model_name,
            "deployed_at":  datetime.now().isoformat(),
            "run_id":        run_id,
            "file":          f"firmware_{firmware_ver}.bin",
            "size_bytes":    file_size,
            "sha256":        sha256,
            "status":        "Production"
        }
        version_path = os.path.join(OTA_FIRMWARE_DIR, "version.json")
        with open(version_path, "w") as f:
            json.dump(version_info, f, indent=2)

        log.info(f"  ✅ Firmware deployed → {bin_dst}")
        log.info(f"  ✅ version.json updated → {firmware_ver}")
        log.info(f"  SHA256: {sha256}  Size: {file_size} bytes")

        # Clean up temp
        shutil.rmtree(local_dir, ignore_errors=True)
        return firmware_ver

    except Exception as e:
        log.error(f"  ❌ OTA deployment failed: {e}", exc_info=True)
        return False

# --------------------
# Main watch loop
# --------------------
def watch():
    log.info("=" * 55)
    log.info("  PFE MLOps - MLflow Production Watcher")
    log.info(f"  MLflow:   {MLFLOW_TRACKING}")
    log.info(f"  OTA dir:  {OTA_FIRMWARE_DIR}")
    log.info(f"  Polling every {POLL_INTERVAL_S}s")
    log.info("=" * 55)

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    client = MlflowClient()
    state  = load_state()

    while True:
        try:
            # Get all registered models
            registered_models = client.search_registered_models()

            for rm in registered_models:
                model_name = rm.name

                # Get all versions in Production stage
                prod_versions = client.search_model_versions(f"name='{model_name}' and current_stage='Production'")

                for mv in prod_versions:
                    version    = mv.version
                    run_id     = mv.run_id
                    state_key  = f"{model_name}_v{version}"

                    # Skip if already processed
                    if state.get(state_key) == "deployed":
                        continue

                    log.info(f"  🔔 New Production model: {model_name} v{version} (run={run_id})")

                    firmware_ver = deploy_to_ota(model_name, version, run_id, client)

                    if firmware_ver:
                        state[state_key] = "deployed"
                        save_state(state)
                        log_ota_event(model_name, firmware_ver, "DEPLOYED",
                                      f"run_id={run_id}")
                        log.info(f"  ✅ State saved: {state_key} → deployed")
                    else:
                        log_ota_event(model_name, str(version), "FAILED",
                                      f"run_id={run_id}")

        except Exception as e:
            log.error(f"Watcher error: {e}", exc_info=True)

        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    watch()
