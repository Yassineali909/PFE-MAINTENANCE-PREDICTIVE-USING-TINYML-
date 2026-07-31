"""
Experience : impact des descripteurs frequentiels sur le MLP.

PROTOCOLE STRICT
  Train      : charges 0 HP et 1 HP
  Validation : charge 2 HP  <- itere sur les features ICI
  Test       : charge 3 HP  <- JAMAIS charge par ce script

Iterer sur les features en regardant la charge 3 HP reintroduirait une fuite
par selection : le score final serait aussi optimiste que le protocole initial.
"""
import os, glob, numpy as np, pandas as pd
from scipy.fft import fft
from scipy.signal import hilbert
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf

DATASET_PATH = os.path.expanduser("~/PFE_IOT/data_csv")
LABELS  = ["ball", "inner_race", "normal", "outer_race"]
WINDOW  = 1024
FS      = 12000.0                       # CWRU Drive End
VAL_SUFFIX, FORBIDDEN = "_2", "_3"

# Roulement SKF 6205-2RS, arbre ~1750 tr/min -> fr = 29.17 Hz
FR   = 1750.0 / 60.0
BPFO = 3.5848 * FR      # ~104.6 Hz  defaut bague externe
BPFI = 5.4152 * FR      # ~158.0 Hz  defaut bague interne
BSF  = 2.3570 * FR      # ~ 68.8 Hz  defaut bille


def load_raw():
    Xtr, ytr, Xva, yva = [], [], [], []
    for ci, label in enumerate(LABELS):
        for f in sorted(glob.glob(os.path.join(DATASET_PATH, label, "*.csv"))):
            src = os.path.basename(f).split("_w")[0]
            if src.endswith(FORBIDDEN):          # garde-fou explicite
                continue
            w = pd.read_csv(f)["vibration"].values[:WINDOW].astype(np.float32)
            if len(w) != WINDOW:
                continue
            if src.endswith(VAL_SUFFIX):
                Xva.append(w); yva.append(ci)
            else:
                Xtr.append(w); ytr.append(ci)
    return (np.array(Xtr, np.float32), np.array(ytr),
            np.array(Xva, np.float32), np.array(yva))


def time_feats(w):
    rms = np.sqrt(np.mean(w**2)); pk = np.max(np.abs(w))
    return [rms, pk, pk/(rms+1e-9), np.mean(w), np.std(w),
            pd.Series(w).skew(), pd.Series(w).kurtosis()]


# --- FS1 : descripteurs actuels (top5 FFT trie par amplitude) -----------
def fs1(w):
    spec = np.abs(fft(w))[:WINDOW//2]
    return np.array(time_feats(w) + list(np.sort(spec)[::-1][:5]), np.float32)


# --- FS2 : energies par bande sur 16 bandes lineaires -------------------
def fs2(w, nb=16):
    spec = np.abs(fft(w))[:WINDOW//2] ** 2
    bands = [np.log1p(b.sum()) for b in np.array_split(spec, nb)]
    return np.array(time_feats(w) + bands, np.float32)


# --- FS3 : spectre d'enveloppe autour des frequences caracteristiques ---
def env_spectrum(w):
    env = np.abs(hilbert(w))
    env = env - env.mean()
    return np.abs(fft(env))[:WINDOW//2] ** 2


def band_energy(spec, f0, df=18.0):
    res = FS / WINDOW                    # ~11.7 Hz par bin
    lo, hi = int((f0-df)/res), int((f0+df)/res) + 1
    lo, hi = max(lo, 1), min(hi, len(spec))
    return np.log1p(spec[lo:hi].sum()) if hi > lo else 0.0


def fs3(w):
    es = env_spectrum(w)
    tot = es[1:].sum() + 1e-9
    feats = time_feats(w)
    for f0 in (BPFO, BPFI, BSF):
        for k in (1, 2, 3):              # fondamentale + 2 harmoniques
            e = band_energy(es, f0 * k)
            feats += [e, e / np.log1p(tot)]   # absolu + relatif
    feats.append(np.log1p(tot))
    return np.array(feats, np.float32)


def build(dim, ncls, seed):
    tf.keras.utils.set_random_seed(seed)
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(ncls, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def evaluate(name, fn, Xtr, ytr, Xva, yva, seeds=(0, 1, 2)):
    Ftr = np.array([fn(w) for w in Xtr], np.float32)
    Fva = np.array([fn(w) for w in Xva], np.float32)
    sc  = StandardScaler().fit(Ftr)
    Ftr, Fva = sc.transform(Ftr), sc.transform(Fva)

    accs, f1s, last = [], [], None
    for s in seeds:
        m = build(Ftr.shape[1], len(LABELS), s)
        m.fit(Ftr, ytr, epochs=50, batch_size=32, verbose=0)
        p = np.argmax(m.predict(Fva, verbose=0), axis=1)
        accs.append(accuracy_score(yva, p))
        f1s.append(f1_score(yva, p, average="macro"))
        last = p
    print("\n=== %s === (%d features)" % (name, Ftr.shape[1]))
    print("  val_accuracy = %.4f +/- %.4f" % (np.mean(accs), np.std(accs)))
    print("  val_f1_macro = %.4f" % np.mean(f1s))
    print("  confusion (dernier seed) :\n", confusion_matrix(yva, last))
    return np.mean(accs)


if __name__ == "__main__":
    print("Chargement (charge 3 HP exclue)...")
    Xtr, ytr, Xva, yva = load_raw()
    print("Train (0,1 HP): %d   Validation (2 HP): %d" % (len(Xtr), len(Xva)))
    print("Frequences caracteristiques : BPFO=%.1f  BPFI=%.1f  BSF=%.1f Hz"
          % (BPFO, BPFI, BSF))
    print("Resolution spectrale : %.1f Hz/bin" % (FS / WINDOW))

    r = {
        "FS1 top5 FFT (actuel)": evaluate("FS1 top5 FFT (actuel)", fs1, Xtr, ytr, Xva, yva),
        "FS2 bandes lineaires":  evaluate("FS2 bandes lineaires",  fs2, Xtr, ytr, Xva, yva),
        "FS3 enveloppe BPFO/BPFI/BSF": evaluate("FS3 enveloppe BPFO/BPFI/BSF", fs3, Xtr, ytr, Xva, yva),
    }
    print("\n" + "="*55)
    for k, v in sorted(r.items(), key=lambda x: -x[1]):
        print("  %-32s %.4f" % (k, v))
    print("\nCharge 3 HP non utilisee. Ne l'evaluer qu'une seule fois, a la fin.")
