# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

PFE (end-of-studies project) IoT/MLOps system for **bearing fault detection** using TinyML on an
ESP32-S3, closing the loop from edge inference back to model retraining and OTA redeployment. The
CWRU bearing dataset (4 classes: `normal`, `ball`, `inner_race`, `outer_race`) is simulated on-device
(no physical vibration sensor is attached — `cwru_signal_sim.h` generates synthetic signal windows).

Data flow, end to end:

```
ESP32-S3 (TFLite Micro: CNN1D + MLP)
   → MQTTS/TLS (EMQX broker, port 8883)
   → mqtt_to_zabbix.py  (Python bridge, subscribes factory/line1/+/telemetry and esp32/ota/status)
        → InfluxDB (esp32_telemetry, ota_events measurements)
        → Zabbix (alerting / trapper metrics)
   → mlops/pipeline.py  (pulls InfluxDB data, trains comparative sklearn models, logs to MLflow)
   → MLflow Model Registry (Staging → Production promotion, backed by SQLite + MinIO S3 artifacts)
   → watcher.py / mlflow_watcher.py  (polls registry for Production promotions)
        → downloads artifact, pushes firmware to ota-server (FastAPI)
   → ESP32 polls GET /firmware/version, downloads GET /firmware/latest → new firmware installed
```

The CNN1D int8 model is the **reference "Production" model** actually flashed to the ESP32 (trained
externally via Edge Impulse, see `mlops/models/`); the sklearn models trained by `pipeline.py`
(RandomForest, ExtraTrees, DecisionTree, NaiveBayes, MLP, XGBoost) are a comparative study logged
alongside it in the same MLflow experiment — they are not what runs on-device.

There are two independent watcher implementations that both close the MLflow → OTA loop
(`mlops/watcher.py` and `mlflow_watcher.py` at repo root); check which one is actually intended to run
before editing — they use different target model names and deployment mechanics (`watcher.py` copies
the artifact directly onto the shared `ota-server/firmware/` volume, `mlflow_watcher.py` uploads it
over HTTP via `POST /firmware/upload`).

## Repository layout

- `docker-compose.yml` — orchestrates the whole backend stack (see Services below).
- `mqtt_to_zabbix.py` — MQTTS subscriber bridging telemetry/OTA events into InfluxDB + Zabbix.
- `mlops/pipeline.py` — fetches sensor data from InfluxDB (falls back to synthetic CWRU-like data if
  empty/unreachable), archives dataset to MinIO, trains/evaluates the comparative sklearn models,
  logs everything to MLflow, and registers models in the MLflow Model Registry.
- `mlops/watcher.py`, `mlflow_watcher.py` — MLflow Production-stage watchers (see above).
- `mlops/models/` — per-model artifacts (`model_v<timestamp>.pkl`, `model_info.json`) plus the
  Edge Impulse export used for the ESP32 build (`edge-impulse-sdk/`, `tflite-model/`,
  `model-parameters/`, `CMakeLists.txt`).
- `ota-server/` — FastAPI firmware delivery server (`main.py`), see endpoints below. `firmware/`
  holds `firmware_latest.bin`, `version.json`, and `history/` (rolled-over prior binaries).
- `esp32_s3_project/` — PlatformIO firmware for the ESP32-S3 (Arduino framework +
  TensorFlowLite_ESP32). `src/main.cpp` runs both CNN1D and MLP inference per cycle for benchmarking,
  publishes telemetry over MQTTS, and simulates CWRU signals via `cwru_signal_sim.h`.
- `certs/` — CA/server TLS certs mounted into EMQX for MQTTS.

## Common commands

### Backend stack (Docker)
```bash
docker compose up -d --build      # bring up emqx, influxdb, zabbix, minio(+init), mlflow, ota-server
docker compose logs -f ota-server  # tail a single service
docker compose down                # stop the stack (add -v to also drop volumes/data)
```
Key ports: EMQX MQTTS `8883`, EMQX dashboard `18083`, InfluxDB `8086`, Zabbix web `8080` / trapper
`10051`, MinIO API `9000` / console `9001`, MLflow UI `5000`, OTA server `8090` (Swagger at
`/docs`).

### MLOps scripts (run from repo root, against the Docker stack)
```bash
python3 mlops/pipeline.py      # train comparative models + log CNN1D reference to MLflow
python3 mlops/watcher.py       # OTA watcher: copies MLflow artifact straight onto firmware volume
python3 mlflow_watcher.py      # OTA watcher: uploads artifact to ota-server via HTTP POST
python3 mqtt_to_zabbix.py      # telemetry bridge (requires broker/InfluxDB/Zabbix reachable)
```
There is no test suite in this repo — verification is done by exercising the running services
(check MLflow UI, query InfluxDB, hit `/firmware/version`, watch ESP32 serial output).

### OTA server (standalone, outside Docker)
```bash
cd ota-server && uvicorn main:app --host 0.0.0.0 --port 8090 --reload
```

### ESP32 firmware (PlatformIO)
```bash
cd esp32_s3_project
pio run                 # build
pio run -t upload       # flash
pio device monitor      # serial monitor (115200 baud)
```
`patch_tflm.py` is a PlatformIO pre-build script that patches a private `operator delete` in the
downloaded `Arduino_TensorFlowLite` library (`compatibility.h`) to public — required for the
TFLite Micro interpreter to link; it's idempotent (checks for a `PATCHED_BY_PFE` marker).

## Things to know when editing

- Credentials/tokens (InfluxDB token, MinIO keys, OTA admin token, MQTT password, Zabbix host) are
  hardcoded as defaults throughout the Python scripts, meant to be overridden via env vars/`.env`
  where `python-dotenv` is used (`pipeline.py`, `mlflow_watcher.py`) — `mqtt_to_zabbix.py` and
  `mlops/watcher.py` do not load `.env` and only read the hardcoded constants at the top of the file.
- `mqtt_client.tls_insecure_set(True)` in `mqtt_to_zabbix.py` and `wifi_secure.setInsecure()` in
  `main.cpp` skip TLS certificate verification — flagged in-code as known technical debt, not an
  oversight to silently "fix" without checking whether the CA chain (`certs/ca.crt`,
  `src/certs/ca_cert.h`) is actually wired up end-to-end first.
- ESP32 arena sizing is hand-tuned (`kArenaSizeCNN = 150*1024`, `kArenaSizeMLP = 40*1024`) and
  falls back from PSRAM to internal RAM if PSRAM isn't detected; a failed tensor allocation halts
  the device in an infinite loop by design (avoids an OTA bootloop) rather than restarting.
- MLflow's backend store is SQLite at `mlops/mlflow.db` (bind-mounted into the container) with
  artifacts in MinIO under `s3://mlflow-artifacts/` — both are gitignored; don't assume they exist
  in a fresh checkout.
- `ota-server/firmware/*.bin` is gitignored (binaries are runtime state, delivered via the watcher
  or manual upload), but `version.json` and `history/` metadata are tracked.
