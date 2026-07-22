import os, glob, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import mlflow
DATASET_PATH="/mnt/hgfs/csv"
LABELS=["ball","inner_race","normal","outer_race"]
WINDOW=1024; TIMESTEPS=32; FEATDIM=32   # 32*32 = 1024, aucune donnee perdue
OUT_TFLITE="/home/yassine/lstm_bearing_int8.tflite"
OUT_KERAS="/home/yassine/lstm_bearing.keras"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("LSTM-BearingFault")
print("Chargement CWRU...")
X_raw, y_raw = [], []
for label in LABELS:
    for f in sorted(glob.glob(os.path.join(DATASET_PATH, label, "*.csv"))):
        w = pd.read_csv(f)["vibration"].values[:WINDOW].astype(np.float32)
        if len(w) == WINDOW:
            X_raw.append(w); y_raw.append(label)
X_raw = np.array(X_raw, dtype=np.float32); y_raw = np.array(y_raw)
print("  Total: %d fenetres" % len(X_raw))
def normalize_window(w):
    return (w - np.mean(w)) / (np.std(w) + 1e-6)
X = np.array([normalize_window(w) for w in X_raw], dtype=np.float32)
X = X.reshape(-1, TIMESTEPS, FEATDIM)   # (N, 32, 32)
le = LabelEncoder(); y = le.fit_transform(y_raw)
assert list(le.classes_) == LABELS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train: %d  Test: %d  shape: %s" % (len(X_train), len(X_test), X_train.shape[1:]))
cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(cw))
def build_lstm(t, fdim, ncls):
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(t, fdim)),
        tf.keras.layers.LSTM(32, return_sequences=False, unroll=True),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(ncls, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m
EPOCHS, BATCH = 25, 64
run = mlflow.start_run(run_name="LSTM_reshape_v1")
mlflow.log_params({"epochs":EPOCHS,"timesteps":TIMESTEPS,"featdim":FEATDIM,"units":32,"norm":"per_window_zscore"})
model = build_lstm(TIMESTEPS, FEATDIM, len(le.classes_))
model.summary()
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_split=0.1, class_weight=class_weight, verbose=1)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
rep = classification_report(y_test, y_pred, target_names=le.classes_, digits=4, output_dict=True)
print("\n"+classification_report(y_test, y_pred, target_names=le.classes_, digits=4))
print("Confusion (Keras):\n", confusion_matrix(y_test, y_pred))
mlflow.log_metrics({"test_accuracy":acc,"f1_macro":rep["macro avg"]["f1-score"]})
model.save(OUT_KERAS)
def rep_data():
    for s in X_train[:300]:
        yield [s.reshape(1, TIMESTEPS, FEATDIM).astype(np.float32)]
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
conv._experimental_lower_tensor_list_ops = False
try:
    tflite_model = conv.convert()
    open(OUT_TFLITE,"wb").write(tflite_model)
    mlflow.log_artifact(OUT_TFLITE)
    print("\nTFLite int8: %.1f KB -> %s" % (len(tflite_model)/1024, OUT_TFLITE))
    CONV_OK = True
except Exception as e:
    print("\n[ECHEC conversion int8 LSTM]:", str(e)[:400]); CONV_OK = False
if CONV_OK:
    it = tf.lite.Interpreter(model_content=tflite_model); it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    s, z = inp["quantization"]
    print("\n[VALIDATION int8] scale=%.6f zp=%d" % (s, z))
    preds=[]
    for x in X_test:
        q = np.clip(np.round(x/s)+z,-128,127).astype(np.int8).reshape(1,TIMESTEPS,FEATDIM)
        it.set_tensor(inp["index"],q); it.invoke()
        preds.append(int(np.argmax(it.get_tensor(out["index"])[0])))
    preds=np.array(preds)
    print("tflite int8 -> acc=%.4f  f1=%.4f" % (accuracy_score(y_test,preds), f1_score(y_test,preds,average="macro")))
    print("Confusion int8:\n", confusion_matrix(y_test, preds))
    mlflow.log_metrics({"tflite_int8_accuracy":accuracy_score(y_test,preds)})
mlflow.end_run(); print("\nTermine LSTM.")
