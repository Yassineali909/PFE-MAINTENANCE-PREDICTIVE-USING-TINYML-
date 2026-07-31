import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser("~/PFE_IOT"))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import mlflow

from cwru_data import load_split, WINDOW, LABELS, TEST_SUFFIX

OUT_TFLITE = "/home/yassine/cnn1d_bearing_int8.tflite"
REGISTERED_NAME = "bearing-fault-cnn1d"   # doit correspondre a TARGET_MODEL_NAME du watcher
OUT_KERAS  = "/home/yassine/cnn1d_bearing.keras"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("CNN1D-BearingFault")

print("Chargement CWRU (split par source, test = charge 3 HP)...")
X_train, y_train, X_test, y_test, le = load_split()
X_train = X_train[..., np.newaxis]
X_test  = X_test[..., np.newaxis]

cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(cw))
print("Class weights:", {LABELS[k]: round(v, 3) for k, v in class_weight.items()})

def build_cnn(input_len, num_classes):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_len, 1)),
        tf.keras.layers.Conv1D(8, 16, strides=2, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Conv1D(16, 8, strides=2, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Conv1D(32, 4, activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

EPOCHS, BATCH = 30, 64
run = mlflow.start_run(run_name="CNN1D_per_source_split")
mlflow.log_params({
    "epochs": EPOCHS, "batch_size": BATCH,
    "normalization": "per_window_zscore", "input_len": WINDOW,
    "validation_protocol": "per_source_split_load3_holdout",
    "test_sources": "B007_3,IR007_3,Normal_3,OR007_at_6_3",
    "n_train": len(X_train), "n_test": len(X_test),
})

model = build_cnn(WINDOW, len(LABELS))
model.summary()
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH,
          validation_split=0.1, class_weight=class_weight, verbose=1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
rep = classification_report(y_test, y_pred, target_names=LABELS, digits=4, output_dict=True)
print("\n[KERAS float32 - charge 3 HP non vue]")
print(classification_report(y_test, y_pred, target_names=LABELS, digits=4))
print("Confusion:\n", confusion_matrix(y_test, y_pred))

mlflow.log_metrics({
    "test_accuracy": acc,
    "f1_macro":  rep["macro avg"]["f1-score"],
    "precision": rep["macro avg"]["precision"],
    "recall":    rep["macro avg"]["recall"],
})
model.save(OUT_KERAS)

def rep_data():
    for s in X_train[:300]:
        yield [s.reshape(1, WINDOW, 1).astype(np.float32)]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.int8
conv.inference_output_type = tf.int8
tflite_model = conv.convert()
with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_model)
print("\nTFLite: %.1f KB -> %s" % (len(tflite_model)/1024, OUT_TFLITE))

it = tf.lite.Interpreter(model_content=tflite_model); it.allocate_tensors()
inp, out = it.get_input_details()[0], it.get_output_details()[0]
s, z   = inp["quantization"]
so, zo = out["quantization"]
print("[VALIDATION int8] input scale=%.6f zp=%d | output scale=%.8f zp=%d" % (s, z, so, zo))
print(">>> Reporter ces 4 valeurs dans main.cpp (constantes de repli TFLM)")

preds = []
for x in X_test:
    q = np.clip(np.round(x/s)+z, -128, 127).astype(np.int8).reshape(1, WINDOW, 1)
    it.set_tensor(inp["index"], q); it.invoke()
    preds.append(int(np.argmax(it.get_tensor(out["index"])[0])))
preds = np.array(preds)
tfl_acc = accuracy_score(y_test, preds)
tfl_f1  = f1_score(y_test, preds, average="macro")
print("\n[TFLITE int8 - charge 3 HP non vue] accuracy=%.4f  f1_macro=%.4f" % (tfl_acc, tfl_f1))
print(classification_report(y_test, preds, target_names=LABELS, digits=4))
print("Confusion (int8):\n", confusion_matrix(y_test, preds))
mlflow.log_metrics({"tflite_int8_accuracy": tfl_acc, "tflite_int8_f1": tfl_f1})

# Parametres de quantification transmis avec le modele : le firmware ne peut
# plus dependre d'une constante compilee des lors que le modele evolue en OTA.
mlflow.set_tags({
    "input_scale":  "%.8f" % s,
    "input_zp":     "%d"   % z,
    "output_scale": "%.8f" % so,
    "output_zp":    "%d"   % zo,
})
mlflow.log_artifact(OUT_TFLITE, artifact_path="model")

# --- Enregistrement dans le Model Registry ---------------------------------
# Le modele entre en Staging, jamais directement en Production.
# La promotion vers Production est un acte humain explicite (Human-in-the-Loop),
# realise depuis l'interface MLflow. C'est cette transition que surveille
# mlflow_watcher.py pour declencher la distribution OTA.
from mlflow.tracking import MlflowClient

client = MlflowClient()
try:
    client.create_registered_model(REGISTERED_NAME)
    print("Modele enregistre cree : %s" % REGISTERED_NAME)
except Exception:
    pass  # deja existant

mv = client.create_model_version(
    name=REGISTERED_NAME,
    source="%s/model" % mlflow.get_artifact_uri(),
    run_id=run.info.run_id,
)
client.transition_model_version_stage(REGISTERED_NAME, mv.version, "Staging")
client.update_model_version(
    name=REGISTERED_NAME, version=mv.version,
    description="CNN 1D int8 — split par source (test charge 3 HP). "
                "accuracy=%.4f f1=%.4f — %.1f KB" % (tfl_acc, tfl_f1, len(tflite_model)/1024),
)
print("\n[REGISTRY] %s version %s -> Staging" % (REGISTERED_NAME, mv.version))
print("[REGISTRY] Promouvoir en Production : http://localhost:5000/#/models/%s" % REGISTERED_NAME)

mlflow.end_run()
print("\nTermine. Si l'accuracy int8 est proche de la float32, la quantification est saine.")
