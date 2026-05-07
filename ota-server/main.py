from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os, json

app = FastAPI(title="PFE OTA Server")

FIRMWARE_DIR = "/firmware"
VERSION_FILE = os.path.join(FIRMWARE_DIR, "version.json")

@app.get("/firmware/version")
def get_version():
    if not os.path.exists(VERSION_FILE):
        return {"version": "0.0.0"}
    with open(VERSION_FILE) as f:
        return json.load(f)

@app.get("/firmware/latest")
def get_firmware():
    bin_files = [f for f in os.listdir(FIRMWARE_DIR) if f.endswith(".bin")]
    if not bin_files:
        raise HTTPException(status_code=404, detail="No firmware available")
    latest = sorted(bin_files)[-1]
    return FileResponse(
        path=os.path.join(FIRMWARE_DIR, latest),
        media_type="application/octet-stream",
        filename=latest
    )

@app.get("/health")
def health():
    return {"status": "ok"}
