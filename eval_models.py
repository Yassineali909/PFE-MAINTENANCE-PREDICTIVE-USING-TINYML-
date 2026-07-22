import os, glob, numpy as np, pandas as pd
from scipy.fft import fft
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import tensorflow as tf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
CSV_ROOT="/mnt/hgfs/csv"
MLP_TFLITE="/home/yassine/mlp_bearing_int8.tflite"
CNN_TFLITE="/home/yassine/PFE_IOT/cnn1d_bearing_int8.tflite"
OUT_DIR=os.path.expanduser("~/PFE_IOT/figures")
CLASSES=["ball","inner_race","normal","outer_race"]; WINDOW=1024
os.makedirs(OUT_DIR, exist_ok=True)
def extract_features(w):
    rms=np.sqrt(np.mean(w**2)); peak=np.max(np.abs(w)); crest=peak/(rms+1e-9)
    mean=np.mean(w); std=np.std(w)
    skew=pd.Series(w).skew(); kurt=pd.Series(w).kurtosis()
    spec=np.abs(fft(w))[:len(w)//2]; top5=np.sort(spec)[::-1][:5]
    return np.concatenate([[rms,peak,crest,mean,std,skew,kurt], top5])
print("Chargement des fenetres CWRU...")
X_raw, y_raw = [], []
for c in CLASSES:
    files = sorted(glob.glob(os.path.join(CSV_ROOT, c, "*.csv")))
    print("  %-12s: %d fenetres" % (c, len(files)))
    for f in files:
        w = pd.read_csv(f)["vibration"].values[:WINDOW].astype(np.float32)
        if len(w) == WINDOW:
            X_raw.append(w); y_raw.append(c)
X_raw = np.array(X_raw); y_raw = np.array(y_raw)
print("  Total: %d fenetres" % len(X_raw))
le = LabelEncoder(); y = le.fit_transform(y_raw)
X_feat = np.array([extract_features(w) for w in X_raw])
idx = np.arange(len(X_raw))
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_feat[idx_train])
y_test = y[idx_test]
print("Jeu de test : %d fenetres" % len(idx_test))
def predict_tflite(path, inputs):
    it = tf.lite.Interpreter(path); it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    s, z = inp["quantization"]; os_, oz = out["quantization"]
    preds = []
    for x in inputs:
        q = np.clip(np.round(x/s)+z, -128, 127).astype(np.int8).reshape(inp["shape"])
        it.set_tensor(inp["index"], q); it.invoke()
        o = it.get_tensor(out["index"])[0].astype(np.float32)
        preds.append(int(np.argmax(o)))
    return np.array(preds)
X_mlp = scaler.transform(X_feat[idx_test])
mlp_pred = predict_tflite(MLP_TFLITE, X_mlp)
cnn_pred = predict_tflite(CNN_TFLITE, X_raw[idx_test])
def save_confusion(y_true, y_pred, title, fname, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=range(4))
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Prediction"); ax.set_ylabel("Vraie classe"); ax.set_title(title)
    thr = cm.max()/2
    for i in range(4):
        for j in range(4):
            v = ("%.2f" % cm[i,j]) if normalize else ("%d" % int(cm[i,j]))
            ax.text(j, i, v, ha="center", va="center",
                    color="white" if cm[i,j] > thr else "black", fontsize=11)
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches="tight")
    plt.close(fig); print("  -> %s" % fname)
def dump_report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    rep = classification_report(y_true, y_pred, labels=range(4),
                                target_names=CLASSES, digits=4, zero_division=0)
    txt = "=== %s ===\nAccuracy: %.4f   F1-macro: %.4f\n\n%s\n" % (name, acc, f1, rep)
    txt += "Confusion (counts):\n" + str(confusion_matrix(y_true, y_pred, labels=range(4))) + "\n"
    with open(os.path.join(OUT_DIR, "rapport_%s.txt" % name.lower()), "w") as f:
        f.write(txt)
    print(txt); return acc, f1
print("="*60)
print("--- CNN 1D (indicatif) ---")
save_confusion(y_test, cnn_pred, "Matrice de confusion - CNN 1D", "confusion_cnn.png")
save_confusion(y_test, cnn_pred, "CNN 1D (normalisee)", "confusion_cnn_norm.png", normalize=True)
dump_report("CNN", y_test, cnn_pred)
print("--- MLP (test legitime) ---")
save_confusion(y_test, mlp_pred, "Matrice de confusion - MLP", "confusion_mlp.png")
save_confusion(y_test, mlp_pred, "MLP (normalisee)", "confusion_mlp_norm.png", normalize=True)
dump_report("MLP", y_test, mlp_pred)
print("Figures dans : %s" % OUT_DIR)

# --- Test permutation ordre de classes CNN ---
from itertools import permutations
print("\n=== Recherche permutation CNN ===")
best=(0,None)
for p in permutations(range(4)):
    remap=np.array(p)[cnn_pred]
    a=accuracy_score(y_test, remap)
    if a>best[0]: best=(a,p)
print("Meilleure permutation:", best[1], "-> accuracy=%.4f" % best[0])
print("Mapping: sortie_CNN -> classe reelle")
for i,pi in enumerate(best[1]):
    print("  indice %d du CNN = %s" % (i, CLASSES[pi]))
