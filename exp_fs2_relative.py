"""
FS2-rel : energies de bande RELATIVES (ratios) au lieu d'absolues.
Hypothese : l'invariance d'echelle est ce qui donne au CNN son transfert
inter-severite. Si vraie, FS2-rel doit rattraper le CNN sur 021".
Protocole : train 007" (0,1,2 HP) -> test 3 HP, puis transfert 014"/021".
"""
import os, glob, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/PFE_IOT"))
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from cwru_data import load_split, LABELS, WINDOW

SEV_DIR = os.path.expanduser("~/PFE_IOT/data_csv_sev")

def feats_rel(w, nb=16):
    rms = np.sqrt(np.mean(w**2)); pk = np.max(np.abs(w))
    # descripteurs temporels SANS dimension (crest, skew, kurt) uniquement
    t = [pk/(rms+1e-9), pd.Series(w).skew(), pd.Series(w).kurtosis()]
    spec = np.abs(fft(w))[:WINDOW//2] ** 2
    tot = spec.sum() + 1e-12
    bands = [b.sum()/tot for b in np.array_split(spec, nb)]   # ratios
    return np.array(t + bands, np.float32)

def load_sev(tag):
    X, y = [], []
    for ci, cls in enumerate(LABELS):
        for f in sorted(glob.glob(os.path.join(SEV_DIR, cls, "*.csv"))):
            if tag not in os.path.basename(f): continue
            w = pd.read_csv(f)["vibration"].values[:WINDOW].astype(np.float32)
            if len(w) == WINDOW: X.append(w); y.append(ci)
    return np.array(X, np.float32), np.array(y)

Xtr, ytr, Xte, yte, le = load_split(verbose=False, normalize=False)
Ftr = np.array([feats_rel(w) for w in Xtr], np.float32)
Fte = np.array([feats_rel(w) for w in Xte], np.float32)
sc = StandardScaler().fit(Ftr)
cw = dict(enumerate(compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)))

tf.keras.utils.set_random_seed(42)
m = tf.keras.Sequential([
    tf.keras.Input(shape=(Ftr.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(LABELS), activation="softmax")])
m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
m.fit(sc.transform(Ftr), ytr, epochs=50, batch_size=32,
      validation_split=0.1, class_weight=cw, verbose=0)

p = np.argmax(m.predict(sc.transform(Fte), verbose=0), axis=1)
print("\n  charge 3 HP (007\")   accuracy = %.4f   [FS2-abs = 1.0000]" % accuracy_score(yte, p))

for tag, ref_cnn, ref_abs in [("014", 0.0726, 0.2654), ("021", 0.9680, 0.6471)]:
    Xs, ys = load_sev(tag)
    ps = np.argmax(m.predict(sc.transform(np.array([feats_rel(w) for w in Xs], np.float32)), verbose=0), axis=1)
    a = accuracy_score(ys, ps)
    print("  severite %s\"        accuracy = %.4f   [CNN = %.4f | FS2-abs = %.4f]"
          % (tag, a, ref_cnn, ref_abs))
    print("    normal predit : %d / %d" % (int((ps == 2).sum()), len(ps)))
