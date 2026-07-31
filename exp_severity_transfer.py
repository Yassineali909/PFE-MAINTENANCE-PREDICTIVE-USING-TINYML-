"""
Transfert inter-severite : modeles entraines sur 007" evalues sur 014" et 021".
Aucun reentrainement. Mesure la limite du domaine de validite.
"""
import os, glob, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/PFE_IOT"))
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from cwru_data import load_split, LABELS, WINDOW, normalize_window

SEV_DIR = os.path.expanduser("~/PFE_IOT/data_csv_sev")
CNN = "/home/yassine/cnn1d_bearing_int8.tflite"
MLP = "/home/yassine/mlp_fs2_bearing_int8.tflite"


def load_sev(tag):
    X, y = [], []
    for ci, cls in enumerate(LABELS):
        for f in sorted(glob.glob(os.path.join(SEV_DIR, cls, "*.csv"))):
            if tag not in os.path.basename(f):
                continue
            w = pd.read_csv(f)["vibration"].values[:WINDOW].astype(np.float32)
            if len(w) == WINDOW:
                X.append(w); y.append(ci)
    return np.array(X, np.float32), np.array(y)


def time_feats(w):
    rms = np.sqrt(np.mean(w**2)); pk = np.max(np.abs(w))
    return [rms, pk, pk/(rms+1e-9), np.mean(w), np.std(w),
            pd.Series(w).skew(), pd.Series(w).kurtosis()]


def fs2(w, nb=16):
    spec = np.abs(fft(w))[:WINDOW//2] ** 2
    return np.array(time_feats(w) + [np.log1p(b.sum())
                    for b in np.array_split(spec, nb)], np.float32)


def run(path, X, shape):
    it = tf.lite.Interpreter(path); it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    s, z = inp["quantization"]
    p = []
    for x in X:
        q = np.clip(np.round(x/s)+z, -128, 127).astype(np.int8).reshape(shape)
        it.set_tensor(inp["index"], q); it.invoke()
        p.append(int(np.argmax(it.get_tensor(out["index"])[0])))
    return np.array(p)


# Scaler du MLP : refit sur le train 007" brut (identique a l'entrainement)
Xtr, _, _, _, _ = load_split(verbose=False, normalize=False)
sc = StandardScaler().fit(np.array([fs2(w) for w in Xtr], np.float32))

for tag in ["014", "021"]:
    Xr, y = load_sev(tag)
    if len(Xr) == 0:
        print("Aucune donnee pour %s\"" % tag); continue
    print("\n" + "="*62)
    print('  SEVERITE %s" — %d fenetres (modeles entraines sur 007" seul)' % (tag, len(Xr)))
    print("="*62)

    pc = run(CNN, np.array([normalize_window(w) for w in Xr], np.float32), (1, WINDOW, 1))
    pm = run(MLP, sc.transform(np.array([fs2(w) for w in Xr], np.float32)), (1, 23))

    for name, p in [("CNN 1D", pc), ("MLP FS2", pm)]:
        print("\n  %-8s accuracy = %.4f" % (name, accuracy_score(y, p)))
        cm = confusion_matrix(y, p, labels=range(4))
        print("           %s" % "  ".join("%-10s" % l[:9] for l in LABELS))
        for i, l in enumerate(LABELS):
            if cm[i].sum() == 0: continue
            print("  %-10s %s" % (l[:9], "  ".join("%-10d" % v for v in cm[i])))
