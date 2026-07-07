"""
OTA Server — FastAPI
PFE IoT | Vantive Tunisia
Port : 8090

Endpoints:
  GET  /                      → health check
  GET  /firmware/version      → {"version": "x.y.z", "model": "CNN1D_int8", ...}
  GET  /firmware/latest       → téléchargement du .bin
  POST /firmware/upload       → upload manuel d'un nouveau .bin (admin)
  GET  /firmware/history      → liste des versions disponibles
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.responses import FileResponse, JSONResponse

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
FIRMWARE_DIR    = Path(os.getenv("FIRMWARE_DIR", "/firmware"))
VERSION_FILE    = FIRMWARE_DIR / "version.json"
FIRMWARE_FILE   = FIRMWARE_DIR / "firmware_latest.bin"
HISTORY_DIR     = FIRMWARE_DIR / "history"

# Simple token pour l'endpoint d'upload (admin)
ADMIN_TOKEN     = os.getenv("OTA_ADMIN_TOKEN")

# --------------------
# Init
# --------------------
FIRMWARE_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="PFE OTA Server",
    description="Firmware delivery server for ESP32 — MLOps loop closing layer",
    version="1.0.0"
)

# --------------------
# Helpers
# --------------------
def load_version_info() -> dict:
    """Charge le fichier version.json, retourne un dict par défaut si absent."""
    if VERSION_FILE.exists():
        try:
            with open(VERSION_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Failed to read version.json: {e}")
    return {
        "version":     "0.0.0",
        "model":       "unknown",
        "accuracy":    None,
        "trained_at":  None,
        "deployed_at": None,
        "sha256":      None,
        "size_bytes":  None,
        "status":      "no_firmware"
    }

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def save_version_info(info: dict):
    with open(VERSION_FILE, "w") as f:
        json.dump(info, f, indent=2)
    log.info(f"version.json updated → version={info.get('version')}")

# --------------------
# Routes
# --------------------

@app.get("/", tags=["Health"])
def health_check():
    """Health check — utilisé par Docker healthcheck et ESP32 au boot."""
    info = load_version_info()
    return {
        "status":          "ok",
        "service":         "PFE OTA Server",
        "current_version": info.get("version", "0.0.0"),
        "firmware_ready":  FIRMWARE_FILE.exists(),
        "timestamp":       datetime.now().isoformat()
    }


@app.get("/firmware/version", tags=["Firmware"])
def get_version():
    """
    Endpoint principal pour l'ESP32.
    L'ESP32 compare sa version locale avec celle retournée ici.
    Si différent → déclenche le téléchargement OTA.
    """
    info = load_version_info()

    if not FIRMWARE_FILE.exists():
        log.warning("GET /firmware/version — no firmware available")
        raise HTTPException(
            status_code=404,
            detail="No firmware available yet. Waiting for MLflow Production promotion."
        )

    log.info(f"GET /firmware/version → {info.get('version')}")
    return info


@app.get("/firmware/latest", tags=["Firmware"])
def download_firmware(x_esp32_version: str = Header(default=None)):
    """
    Téléchargement du firmware .bin.
    L'ESP32 envoie son header X-ESP32-Version pour logging.
    """
    if not FIRMWARE_FILE.exists():
        log.warning("GET /firmware/latest — firmware not found")
        raise HTTPException(
            status_code=404,
            detail="No firmware binary available. Deploy a model to Production first."
        )

    info    = load_version_info()
    version = info.get("version", "unknown")

    log.info(
        f"GET /firmware/latest → serving v{version} "
        f"(requested by ESP32 v{x_esp32_version or 'unknown'})"
    )

    return FileResponse(
        path=str(FIRMWARE_FILE),
        media_type="application/octet-stream",
        filename=f"firmware_v{version}.bin",
        headers={
            "X-Firmware-Version": version,
            "X-Model-Name":       info.get("model", "unknown"),
            "X-SHA256":           info.get("sha256", ""),
        }
    )


@app.post("/firmware/upload", tags=["Admin"])
async def upload_firmware(
    version:     str,
    model_name:  str       = "CNN1D_int8",
    accuracy:    float     = None,
    trained_at:  str       = None,
    file:        UploadFile = File(...),
    x_admin_token: str     = Header(default=None)
):
    """
    Upload manuel d'un firmware .bin.
    Utilisé par le watcher MLflow pour déployer automatiquement
    un modèle promu en Production.

    Header requis : X-Admin-Token
    """
    # Auth simple
    if x_admin_token != ADMIN_TOKEN:
        log.warning(f"POST /firmware/upload — unauthorized attempt (version={version})")
        raise HTTPException(status_code=401, detail="Invalid admin token")

    # Validation fichier
    if not file.filename.endswith(".bin"):
        raise HTTPException(status_code=400, detail="Only .bin files accepted")

    # Backup de l'ancien firmware dans history/
    if FIRMWARE_FILE.exists():
        old_info    = load_version_info()
        old_version = old_info.get("version", "unknown")
        backup_path = HISTORY_DIR / f"firmware_v{old_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"
        FIRMWARE_FILE.rename(backup_path)
        log.info(f"Old firmware backed up → {backup_path.name}")

    # Sauvegarde du nouveau .bin
    content = await file.read()
    with open(FIRMWARE_FILE, "wb") as f:
        f.write(content)

    sha256 = compute_sha256(FIRMWARE_FILE)
    size   = len(content)

    # Mise à jour version.json
    info = {
        "version":     version,
        "model":       model_name,
        "accuracy":    accuracy,
        "trained_at":  trained_at,
        "deployed_at": datetime.now().isoformat(),
        "sha256":      sha256,
        "size_bytes":  size,
        "status":      "Production"
    }
    save_version_info(info)

    log.info(
        f"POST /firmware/upload → v{version} deployed "
        f"({size/1024:.1f} KB, sha256={sha256[:16]}...)"
    )

    return {
        "message":    f"Firmware v{version} deployed successfully",
        "version":    version,
        "size_kb":    round(size / 1024, 1),
        "sha256":     sha256,
        "deployed_at": info["deployed_at"]
    }


@app.get("/firmware/history", tags=["Admin"])
def get_firmware_history():
    """
    Liste toutes les versions archivées dans history/.
    Utile pour audit et rollback manuel.
    """
    if not HISTORY_DIR.exists():
        return {"history": [], "count": 0}

    files = sorted(HISTORY_DIR.glob("*.bin"), key=lambda f: f.stat().st_mtime, reverse=True)
    history = [
        {
            "filename":     f.name,
            "size_kb":      round(f.stat().st_size / 1024, 1),
            "archived_at":  datetime.fromtimestamp(f.stat().st_mtime).isoformat()
        }
        for f in files
    ]

    current = load_version_info()
    return {
        "current_version": current.get("version"),
        "history":         history,
        "count":           len(history)
    }
