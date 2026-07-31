"""
Evaluation FINALE de FS2 sur la charge 3 HP. A n'executer qu'une fois.
FS2 a ete selectionne sur la charge 2 HP (validation), sans jamais voir 3 HP.
Protocole identique au CNN 1D : train 0+1+2 HP, test 3 HP.
"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/PFE_IOT"))
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
import tensorflow as tf
import mlflow
from cwru_data import load_split, LABELS, WINDOW

OUT_TFLITE = "/home/yassine/mlp_fs2_bearing_int8.tflite"

def time_feats(w):
    rms = np.sqrt(np.mean(w**2)); pk = np.max(np.abs(w))
    return [rms, pk, pk/(rms+1e-9), np.mean(w), np.std(w),
            pd.Series(w).skew(), pd.Series(w).kurtosis()]

def fs2(w, nb=16):
    spec = np.abs(fft(w))[:WINDOW//2] ** 2
    return np.array(time_feats(w) + [np.log1p(b.sum())
                    for b in np.array_split(spec, nb)], np.float32)

print("Chargement (signal brut, split par source)...")
Xtr, ytr, Xte, yte, le = load_split(normalize=False)
Ftr = np.array([fs2(w) for w in Xtr], np.float32)
Fte = np.array([fs2(w) for w in Xte], np.float32)
sc = StandardScaler().fit(Ftr)
Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
print("Train: %d  Test: %d  features: %d" % (len(Ftr), len(Fte), Ftr.shape[1]))

cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
class_weight = dict(enumerate(cw))

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLP-BearingFault")
mlflow.start_run(run_name="MLP_FS2_band_energies")
mlflow.log_params({
    "features": "7_temporels + 16_energies_bandes_log",
    "n_features": int(Ftr.shape[1]),
    "feature_selection": "validee_sur_charge_2HP",
    "validation_protocol": "per_source_split_load3_holdout",
})

tf.keras.utils.set_random_seed(42)
m = tf.keras.Sequential([
    tf.keras.Input(shape=(Ftr.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(LABELS), activation="softmax"),
])
m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
m.fit(Ftr, ytr, epochs=50, batch_size=32, validation_split=0.1,
      class_weight=class_weight, verbose=0)

def rep_data():
    for s in Ftr[:200]:
        yield [s.reshape(1, -1).astype(np.float32)]

conv = tf.lite.TFLiteConverter.from_keras_model(m)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.int8
conv.inference_output_type = tf.int8
tfl = conv.convert()
open(OUT_TFLITE, "wb").write(tfl)

it = tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
inp, out = it.get_input_details()[0], it.get_output_details()[0]
s, z = inp["quantization"]
preds = []
for x in Fte:
    q = np.clip(np.round(x/s)+z, -128, 127).astype(np.int8).reshape(1, -1)
    it.set_tensor(inp["index"], q); it.invoke()
    preds.append(int(np.argmax(it.get_tensor(out["index"])[0])))
preds = np.array(preds)
acc = accuracy_score(yte, preds); f1 = f1_score(yte, preds, average="macro")

print("\n" + "="*58)
print("  MLP FS2 int8 — charge 3 HP jamais vue")
print("  TFLite : %.1f KB   (FS1 = 8.1 KB, CNN 1D = 14.8 KB)" % (len(tfl)/1024))
print("  accuracy = %.4f   f1_macro = %.4f" % (acc, f1))
print("="*58)
print(classification_report(yte, preds, target_names=LABELS, digits=4))
print("Confusion:\n", confusion_matrix(yte, preds))
mlflow.log_metrics({"tflite_int8_accuracy": acc, "tflite_int8_f1": f1,
                    "tflite_size_kb": len(tfl)/1024})
mlflow.log_artifact(OUT_TFLITE, artifact_path="model")
mlflow.end_run()
