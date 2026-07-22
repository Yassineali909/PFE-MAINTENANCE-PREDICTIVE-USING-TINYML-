"""
mlflow_watcher.py
PFE IoT | Vantive Tunisia

Rôle : Fermer la boucle MLOps
  MLflow Model Registry (Production) → OTA Server → ESP32

Logique :
  1. Polling toutes les X secondes sur MLflow Model Registry
  2. Détecte quand un modèle passe en stage "Production"
  3. Télécharge l'artifact .bin depuis MinIO via MLflow
  4. Uploade le .bin vers l'OTA Server via POST /firmware/upload
  5. Log l'événement dans InfluxDB (measurement: ota_events)

Usage :
  python mlflow_watcher.py

Variables d'environnement (ou .env) :
  MLFLOW_TRACKING       → http://localhost:5000
  OTA_SERVER_URL        → http://localhost:8090
  OTA_ADMIN_TOKEN       → pfe_ota_admin_2024
  POLL_INTERVAL_SEC     → 30
  INFLUXDB_URL          → http://localhost:8086
  INFLUXDB_TOKEN        → ...
  INFLUXDB_ORG          → stage_pfe
  INFLUXDB_BUCKET       → esp32-sensor
"""

import os
import time
import logging
import requests
import tempfile
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import mlflow
from mlflow.tracking import MlflowClient

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

load_dotenv()

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
MLFLOW_TRACKING   = os.getenv("MLFLOW_TRACKING",   "http://localhost:5000")
OTA_SERVER_URL    = os.getenv("OTA_SERVER_URL",     "http://localhost:8090")
OTA_ADMIN_TOKEN   = os.getenv("OTA_ADMIN_TOKEN",    "pfe_ota_admin_2024")
POLL_INTERVAL     = int(os.getenv("POLL_INTERVAL_SEC", "30"))

INFLUXDB_URL      = os.getenv("INFLUXDB_URL",    "http://localhost:8086")
INFLUXDB_TOKEN    = os.getenv("INFLUXDB_TOKEN",  "")
INFLUXDB_ORG      = os.getenv("INFLUXDB_ORG",    "stage_pfe")
INFLUXDB_BUCKET   = os.getenv("INFLUXDB_BUCKET", "esp32-sensor")

# Modèle ciblé dans le registry MLflow
# On surveille uniquement le CNN1D — modèle de référence Production
TARGET_MODEL_NAME = "bearing-fault-cnn1d"

# --------------------
# State — versions déjà déployées (évite les re-déploiements)
# --------------------
deployed_versions: set = set()

# --------------------
# InfluxDB client
# --------------------
def get_influx_write_api():
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    return client, client.write_api(write_options=SYNCHRONOUS)

def log_ota_event_to_influx(version: str, status: str, model_name: str, accuracy: float = None):
    """Log l'événement de déploiement OTA dans InfluxDB."""
    try:
        client, write_api = get_influx_write_api()
        p = (
            Point("ota_deployments")
            .tag("source", "mlflow_watcher")
            .tag("model",  model_name)
            .tag("status", status)
            .field("version",  version)
            .field("accuracy", float(accuracy) if accuracy else 0.0)
            .time(int(time.time()), WritePrecision.S)
        )
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=p)
        write_api.close()
        client.close()
        log.info(f"InfluxDB: OTA deployment event logged (v{version}, {status})")
    except Exception as e:
        log.warning(f"InfluxDB log failed: {e}")

# --------------------
# MLflow helpers
# --------------------
def get_production_versions(client: MlflowClient, model_name: str) -> list:
    """Retourne la liste des versions en stage Production pour un modèle."""
    try:
        versions = client.search_model_versions(f"name='{model_name}' and current_stage='Production'")
        return versions
    except mlflow.exceptions.RestException:
        # Modèle pas encore créé dans le registry
        return []
    except Exception as e:
        log.warning(f"MLflow registry error for '{model_name}': {e}")
        return []

def download_bin_artifact(client: MlflowClient, run_id: str, tmp_dir: str) -> Path | None:
    """
    Télécharge l'artifact .bin depuis MLflow/MinIO.
    Retourne le chemin local du .bin, ou None si introuvable.
    """
    try:
        artifacts = client.list_artifacts(run_id)
        bin_artifact = None

        for art in artifacts:
            if art.path.endswith(".bin"):
                bin_artifact = art.path
                break

        if not bin_artifact:
            log.warning(f"No .bin artifact found in run {run_id}")
            return None

        local_path = client.download_artifacts(run_id, bin_artifact, tmp_dir)
        log.info(f"Artifact downloaded → {local_path}")
        return Path(local_path)

    except Exception as e:
        log.error(f"Failed to download artifact from run {run_id}: {e}")
        return None

# --------------------
# OTA Server upload
# --------------------
def upload_to_ota_server(bin_path: Path, version: str, model_name: str, accuracy: float = None) -> bool:
    """
    Uploade le .bin vers l'OTA Server via POST /firmware/upload.
    Retourne True si succès.
    """
    url = f"{OTA_SERVER_URL}/firmware/upload"
    params = {
        "version":    version,
        "model_name": model_name,
        "accuracy":   accuracy,
        "trained_at": datetime.now().isoformat()
    }
    headers = {
        "X-Admin-Token": OTA_ADMIN_TOKEN
    }

    try:
        with open(bin_path, "rb") as f:
            response = requests.post(
                url,
                params=params,
                headers=headers,
                files={"file": (bin_path.name, f, "application/octet-stream")},
                timeout=30
            )

        if response.status_code == 200:
            data = response.json()
            log.info(
                f"OTA Server: firmware uploaded ✅ "
                f"v{data['version']} ({data['size_kb']} KB) "
                f"sha256={data['sha256'][:16]}..."
            )
            return True
        else:
            log.error(f"OTA Server upload failed: {response.status_code} — {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        log.error(f"OTA Server not reachable at {OTA_SERVER_URL}")
        return False
    except Exception as e:
        log.error(f"Upload error: {e}")
        return False

# --------------------
# Core polling loop
# --------------------
def check_and_deploy(client: MlflowClient):
    """
    Vérifie les nouvelles versions Production dans MLflow.
    Déploie si une version non encore déployée est trouvée.
    """
    prod_versions = get_production_versions(client, TARGET_MODEL_NAME)

    if not prod_versions:
        log.debug(f"No Production version found for '{TARGET_MODEL_NAME}'")
        return

    for mv in prod_versions:
        version_key = f"{mv.name}@v{mv.version}"

        if version_key in deployed_versions:
            log.debug(f"Already deployed: {version_key}")
            continue

        log.info(f"🚀 New Production version detected: {version_key} (run_id={mv.run_id})")

        # Récupère les métriques du run MLflow
        run = client.get_run(mv.run_id)
        accuracy = run.data.metrics.get("accuracy")
        trained_at = run.data.params.get("trained_at", datetime.now().isoformat())

        # Version string pour l'OTA (ex: "1.0.3")
        ota_version = f"1.0.{mv.version}"

        # Téléchargement du .bin
        with tempfile.TemporaryDirectory() as tmp_dir:
            bin_path = download_bin_artifact(client, mv.run_id, tmp_dir)

            if bin_path is None:
                log.warning(
                    f"No .bin artifact for {version_key}. "
                    f"Make sure Edge Impulse .bin is logged as MLflow artifact."
                )
                # Marque quand même comme traité pour ne pas boucler
                deployed_versions.add(version_key)
                log_ota_event_to_influx(ota_version, "NO_BIN_ARTIFACT", TARGET_MODEL_NAME, accuracy)
                continue

            # Upload vers OTA Server
            success = upload_to_ota_server(bin_path, ota_version, TARGET_MODEL_NAME, accuracy)

        if success:
            deployed_versions.add(version_key)
            log_ota_event_to_influx(ota_version, "DEPLOYED", TARGET_MODEL_NAME, accuracy)
            log.info(
                f"✅ Pipeline closed: MLflow Production → OTA Server v{ota_version} "
                f"(acc={accuracy:.4f if accuracy else 'N/A'})"
            )
        else:
            log_ota_event_to_influx(ota_version, "UPLOAD_FAILED", TARGET_MODEL_NAME, accuracy)
            log.error(f"❌ Deployment failed for {version_key}")


def main():
    log.info("=" * 60)
    log.info("  MLflow Watcher — OTA Deployment Agent")
    log.info(f"  MLflow    : {MLFLOW_TRACKING}")
    log.info(f"  OTA Server: {OTA_SERVER_URL}")
    log.info(f"  Target    : {TARGET_MODEL_NAME}")
    log.info(f"  Polling   : every {POLL_INTERVAL}s")
    log.info("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING)

    # Charge les versions déjà en Production au démarrage
    # (évite de re-déployer au restart du watcher)
    existing = get_production_versions(client, TARGET_MODEL_NAME)
    for mv in existing:
        deployed_versions.add(f"{mv.name}@v{mv.version}")
    if deployed_versions:
        log.info(f"Already deployed versions (skipped): {deployed_versions}")

    log.info("Watcher started — waiting for Production promotions...")

    while True:
        try:
            check_and_deploy(client)
        except KeyboardInterrupt:
            log.info("Watcher stopped by user.")
            break
        except Exception as e:
            log.error(f"Unexpected error in polling loop: {e}", exc_info=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
