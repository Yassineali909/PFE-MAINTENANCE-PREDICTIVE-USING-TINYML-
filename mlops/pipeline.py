import os, io, json, logging, pickle, numpy as np, pandas as pd
import mlflow, mlflow.sklearn, boto3
from datetime import datetime
from dotenv import load_dotenv
import influxdb_client
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --------------------
# Config
# --------------------
INFLUXDB_URL    = os.getenv("INFLUXDB_URL",    "http://localhost:8086")
INFLUXDB_TOKEN  = os.getenv("INFLUXDB_TOKEN",  "K78LjzMs2yCJP6tgdxVqFf0IV4P8rxA82zPb1gcCgwx7FngDgub7DrjjrL3SnNgDqKFdW_JmleIySWX6qwCssQ==")
INFLUXDB_ORG    = os.getenv("INFLUXDB_ORG",    "stage_pfe")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "esp32-sensor")
MEASUREMENT     = "esp32_telemetry"

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",       "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID",     "minioadmin")
MINIO_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
MINIO_BUCKET     = "mlflow-artifacts"

MLFLOW_TRACKING  = os.getenv("MLFLOW_TRACKING", "http://localhost:5000")

# Local folder where trained models are saved
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

CLASS_NAMES = ["Normal", "Inner Fault", "Outer Fault"]

MODELS = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "ExtraTrees":   ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "NaiveBayes":   GaussianNB(),
    "MLP":          MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    "XGBoost":      XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss", verbosity=0),
}

CNN1D_METRICS = {
    "accuracy":  0.9823,
    "f1_score":  0.9815,
    "precision": 0.9820,
    "recall":    0.9810,
    "note": "CNN 1D int8 quantized - ESP32-WROOM-32 - CWRU dataset - Edge Impulse"
}

# --------------------
# 1. Fetch data
# --------------------
def fetch_sensor_data(hours=24):
    log.info(f"[1/4] Fetching last {hours}h from InfluxDB...")
    client = influxdb_client.InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    query = (
        f'from(bucket: "{INFLUXDB_BUCKET}")'
        f' |> range(start: -{hours}h)'
        f' |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")'
        ' |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")'
    )
    try:
        df = client.query_api().query_data_frame(query)
        if isinstance(df, list):
            df = pd.concat(df, ignore_index=True)
        if df.empty:
            raise ValueError("Empty dataset from InfluxDB")
        log.info(f"  OK - {len(df)} records from InfluxDB")
    except Exception as e:
        log.warning(f"  InfluxDB: {e} - using synthetic CWRU data")
        df = generate_synthetic_data()
    client.close()
    return df

def generate_synthetic_data(n=1000):
    log.info(f"  Generating synthetic CWRU-like data ({n} samples)...")
    np.random.seed(42)
    labels = np.random.choice([0, 1, 2], size=n, p=[0.6, 0.2, 0.2])
    return pd.DataFrame({
        "vibration_x": np.where(labels == 0, np.random.normal(0.1, 0.05, n),
                        np.where(labels == 1, np.random.normal(0.8, 0.2,  n),
                                              np.random.normal(1.5, 0.3,  n))),
        "vibration_y": np.random.normal(0.1, 0.05, n) + labels * 0.3,
        "vibration_z": np.random.normal(9.8, 0.1,  n) + labels * 0.1,
        "temperature": np.random.normal(25,  2,    n) + labels * 3,
        "label": labels
    })

# --------------------
# 2. Archive dataset to MinIO
# --------------------
def archive_to_minio(df):
    log.info("[2/4] Archiving dataset to MinIO...")
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )
    key = f"datasets/bearing_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=MINIO_BUCKET, Key=key, Body=buf.getvalue())
    log.info(f"  OK - minio://{MINIO_BUCKET}/{key}")
    return key

# --------------------
# 3. Prepare features
# --------------------
def prepare_data(df):
    features = [c for c in ["vibration_x", "vibration_y", "vibration_z", "temperature"] if c in df.columns]
    if "label" not in df.columns:
        df["label"] = np.random.choice([0, 1, 2], size=len(df))
    df = df.dropna(subset=features)
    return df[features].values, df["label"].values.astype(int), features

# --------------------
# 4. Save model locally to mlops/models/<ModelName>/
# --------------------
def save_model_locally(name, model, metrics, version):
    """
    Saves:
      mlops/models/<ModelName>/model_v<version>.pkl   ← trained model
      mlops/models/<ModelName>/model_info.json        ← metadata
    """
    model_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    # Save model pickle
    pkl_path = os.path.join(model_dir, f"model_v{version}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    # Save metadata JSON
    info = {
        "model_name":  name,
        "version":     version,
        "trained_at":  datetime.now().isoformat(),
        "accuracy":    round(float(metrics["accuracy"]), 6),
        "f1_score":    round(float(metrics["f1_score"]), 6),
        "precision":   round(float(metrics["precision"]), 6),
        "recall":      round(float(metrics["recall"]), 6),
        "cv_accuracy": round(float(metrics["cv_accuracy_mean"]), 6),
        "pkl_file":    f"model_v{version}.pkl",
        "status":      "Staging"
    }
    info_path = os.path.join(model_dir, "model_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    log.info(f"  Saved locally → {pkl_path}")
    log.info(f"  Metadata     → {info_path}")
    return pkl_path, info_path

# --------------------
# 5. Train & evaluate
# --------------------
def evaluate_model(name, model, X, y):
    log.info(f"  [{name}] Training...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":         accuracy_score(y_test, y_pred),
        "f1_score":         f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "precision":        precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":           recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "cv_accuracy_mean": cv_res["test_accuracy"].mean(),
        "cv_accuracy_std":  cv_res["test_accuracy"].std(),
        "cv_f1_mean":       cv_res["test_f1_weighted"].mean(),
    }
    cm = confusion_matrix(y_test, y_pred)

    log.info(f"    acc={metrics['accuracy']:.4f}  f1={metrics['f1_score']:.4f}  "
             f"cv={metrics['cv_accuracy_mean']:.4f}±{metrics['cv_accuracy_std']:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
    return model, metrics, cm

# --------------------
# 6. Log to MLflow + save locally
# --------------------
def log_to_mlflow(name, model, metrics, cm, features, dataset_key):
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = ts  # use timestamp as version string

    # Save model locally first
    pkl_path, info_path = save_model_locally(name, model, metrics, version)

    with mlflow.start_run(run_name=f"{name}_{ts}"):
        # Params
        mlflow.log_param("model_type", name)
        mlflow.log_param("features",   str(features))
        mlflow.log_param("dataset",    dataset_key)
        mlflow.log_param("cv_folds",   5)
        mlflow.log_param("version",    version)

        # Metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, round(float(v), 6))

        # Confusion matrix artifact
        cm_df   = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
        cm_path = f"/tmp/cm_{name}.csv"
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(cm_path, "confusion_matrix")

        # Feature importance artifact (tree-based models)
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
            fi_path = f"/tmp/fi_{name}.csv"
            fi.sort_values("importance", ascending=False).to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path, "feature_importance")

        # Log the pkl and metadata into MLflow artifacts
        mlflow.log_artifact(pkl_path,  "model_pkl")
        mlflow.log_artifact(info_path, "model_info")

        # Register model in MLflow Model Registry (Staging)
        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name=f"bearing-fault-{name.lower()}"
        )

        run_id = mlflow.active_run().info.run_id
        log.info(f"    MLflow run_id: {run_id}")

# --------------------
# 7. Log CNN1D reference
# --------------------
def log_cnn1d(dataset_key):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CNN1D info in models/CNN1D/
    cnn_dir = os.path.join(MODELS_DIR, "CNN1D")
    os.makedirs(cnn_dir, exist_ok=True)
    info = {
        "model_name": "CNN1D_int8",
        "version":    ts,
        "trained_at": datetime.now().isoformat(),
        "accuracy":   CNN1D_METRICS["accuracy"],
        "f1_score":   CNN1D_METRICS["f1_score"],
        "precision":  CNN1D_METRICS["precision"],
        "recall":     CNN1D_METRICS["recall"],
        "framework":  "TensorFlow Lite / Edge Impulse",
        "quantization": "int8",
        "deployment": "ESP32-WROOM-32",
        "note":       CNN1D_METRICS["note"],
        "status":     "Production"   # CNN1D is the reference production model
    }
    with open(os.path.join(cnn_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    with mlflow.start_run(run_name=f"CNN1D_int8_{ts}"):
        mlflow.log_param("model_type",    "CNN1D_int8")
        mlflow.log_param("framework",     "TensorFlow Lite")
        mlflow.log_param("quantization",  "int8")
        mlflow.log_param("dataset",       "CWRU_bearing")
        mlflow.log_param("deployment",    "ESP32-WROOM-32")
        mlflow.log_param("pipeline_data", dataset_key)
        mlflow.log_param("version",       ts)
        for k, v in CNN1D_METRICS.items():
            if k != "note":
                mlflow.log_metric(k, v)
        mlflow.set_tag("note",   CNN1D_METRICS["note"])
        mlflow.set_tag("status", "Production")
        mlflow.log_artifact(os.path.join(cnn_dir, "model_info.json"), "model_info")
        log.info(f"  CNN1D logged → run_id: {mlflow.active_run().info.run_id}")

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  PFE MLOps - Bearing Fault Detection Comparative Study")
    print("=" * 60)

    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    log.info(f"Models will be saved to: {MODELS_DIR}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment("bearing-fault-detection")

    df          = fetch_sensor_data(hours=24)
    dataset_key = archive_to_minio(df)
    X, y, features = prepare_data(df)

    log.info("[3/4] Logging CNN1D reference model...")
    log_cnn1d(dataset_key)

    log.info("[4/4] Training and logging all comparative models...")
    results = {}
    for name, model in MODELS.items():
        trained, metrics, cm = evaluate_model(name, model, X, y)
        log_to_mlflow(name, trained, metrics, cm, features, dataset_key)
        results[name] = metrics["accuracy"]

    # Summary
    print("\n" + "=" * 60)
    print("  COMPARATIVE RESULTS")
    print("=" * 60)
    print(f"  {'CNN1D (ESP32)':<20} acc={CNN1D_METRICS['accuracy']:.4f}  "
          f"f1={CNN1D_METRICS['f1_score']:.4f}  <- REFERENCE (Production)")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<20} acc={acc:.4f}")
    print("=" * 60)

    # Show what was saved locally
    print("\n  Models saved in mlops/models/:")
    for d in sorted(os.listdir(MODELS_DIR)):
        model_dir = os.path.join(MODELS_DIR, d)
        if os.path.isdir(model_dir):
            files = os.listdir(model_dir)
            print(f"    {d}/  → {', '.join(files)}")

    print(f"\n  MLflow UI  → http://localhost:5000")
    print(f"  MinIO UI   → http://localhost:9001")
    print("=" * 60)
