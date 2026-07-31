"""
Chargement CWRU avec partitionnement par source (charge moteur).
Train : charges 0, 1, 2 HP  |  Test : charge 3 HP, jamais vue.

Justification : les fenetres consecutives se recouvrent a 50 %. Un decoupage
aleatoire place des echantillons quasi identiques de part et d'autre du split
et produit une accuracy artificiellement elevee.
"""
import os, glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = os.path.expanduser("~/PFE_IOT/data_csv")
LABELS       = ["ball", "inner_race", "normal", "outer_race"]
WINDOW       = 1024
TEST_SUFFIX  = "_3"
SEED         = 42


def source_of(path):
    return os.path.basename(path).split("_w")[0]


def normalize_window(w):
    return (w - np.mean(w)) / (np.std(w) + 1e-6)


def load_split(verbose=True):
    """Retourne Xtr, ytr, Xte, yte, label_encoder. X normalises, shape (N, 1024)."""
    Xtr_raw, ytr_raw, Xte_raw, yte_raw = [], [], [], []
    src_train, src_test = set(), set()

    for label in LABELS:
        files = sorted(glob.glob(os.path.join(DATASET_PATH, label, "*.csv")))
        if not files:
            raise FileNotFoundError("Aucun CSV dans %s/%s" % (DATASET_PATH, label))
        n_tr = n_te = 0
        for f in files:
            w = pd.read_csv(f)["vibration"].values[:WINDOW].astype(np.float32)
            if len(w) != WINDOW:
                continue
            src = source_of(f)
            if src.endswith(TEST_SUFFIX):
                Xte_raw.append(w); yte_raw.append(label); src_test.add(src); n_te += 1
            else:
                Xtr_raw.append(w); ytr_raw.append(label); src_train.add(src); n_tr += 1
        if verbose:
            print("  %-12s : %4d train / %4d test" % (label, n_tr, n_te))

    inter = src_train & src_test
    assert not inter, "FUITE : sources communes train/test -> %s" % inter
    assert len(Xte_raw) > 0, "Jeu de test vide : verifier TEST_SUFFIX"

    le = LabelEncoder(); le.fit(LABELS)
    assert list(le.classes_) == LABELS, "Ordre LabelEncoder inattendu"

    Xtr = np.array([normalize_window(w) for w in Xtr_raw], dtype=np.float32)
    Xte = np.array([normalize_window(w) for w in Xte_raw], dtype=np.float32)
    ytr = le.transform(ytr_raw)
    yte = le.transform(yte_raw)

    perm = np.random.RandomState(SEED).permutation(len(Xtr))
    Xtr, ytr = Xtr[perm], ytr[perm]

    if verbose:
        print("  Sources train : %s" % sorted(src_train))
        print("  Sources test  : %s" % sorted(src_test))
        print("  TOTAL : %d train / %d test" % (len(Xtr), len(Xte)))
    return Xtr, ytr, Xte, yte, le


def load_features_split(verbose=True):
    """Variante 12 features pour le MLP. Scaler ajuste sur le TRAIN uniquement."""
    from scipy.fft import fft
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr, Xte, yte, le = load_split(verbose=verbose)

    def feats(w):
        rms = np.sqrt(np.mean(w**2)); pk = np.max(np.abs(w))
        crest = pk / (rms + 1e-9)
        sk = pd.Series(w).skew(); ku = pd.Series(w).kurtosis()
        spec = np.abs(fft(w))[:len(w)//2]
        top5 = np.sort(spec)[::-1][:5]
        return np.concatenate([[rms, pk, crest, np.mean(w), np.std(w), sk, ku], top5])

    Ftr = np.array([feats(w) for w in Xtr], dtype=np.float32)
    Fte = np.array([feats(w) for w in Xte], dtype=np.float32)
    scaler = StandardScaler().fit(Ftr)
    return scaler.transform(Ftr), ytr, scaler.transform(Fte), yte, le, scaler
