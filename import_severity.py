"""
Conversion des .mat CWRU 014" et 021" vers CSV fenetres.
Sortie : ~/PFE_IOT/data_csv_sev/<classe>/<SOURCE>_w<n>.csv
Convention de nommage identique a data_csv/ pour reutiliser cwru_data.py.
"""
import os, sys, glob
import scipy.io, numpy as np, pandas as pd

RAW_DIR = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else "."
CSV_DIR = os.path.expanduser("~/PFE_IOT/data_csv_sev")
WINDOW, OVERLAP = 1024, 0.5


def label_of(stem):
    if stem.startswith("IR"): return "inner_race"
    if stem.startswith("OR"): return "outer_race"
    if stem.startswith("B"):  return "ball"
    if stem.lower().startswith("normal"): return "normal"
    raise ValueError("Classe indeterminee : " + stem)


def de_time(mat):
    for k in mat:
        if "DE_time" in k:
            return mat[k].flatten()
    raise KeyError("DE_time introuvable")


def segment(sig, w, ov):
    step = int(w * (1 - ov))
    return [sig[i:i+w] for i in range(0, len(sig) - w + 1, step)]


files = sorted(glob.glob(os.path.join(RAW_DIR, "*.mat")))
if not files:
    sys.exit("Aucun .mat dans %s" % RAW_DIR)

total = 0
for fp in files:
    stem = os.path.splitext(os.path.basename(fp))[0].replace("@", "_at_")
    label = label_of(stem)
    os.makedirs(os.path.join(CSV_DIR, label), exist_ok=True)
    sig = de_time(scipy.io.loadmat(fp))
    wins = segment(sig, WINDOW, OVERLAP)
    ku = pd.Series(wins[0]).kurtosis() if wins else float("nan")
    for i, w in enumerate(wins):
        pd.DataFrame({"vibration": w}).to_csv(
            os.path.join(CSV_DIR, label, "%s_w%d.csv" % (stem, i)), index=False)
    total += len(wins)
    print("  %-18s -> %-11s %4d fenetres  kurtosis[0]=%+.2f"
          % (stem, label, len(wins), ku))

print("\nTotal : %d fenetres dans %s" % (total, CSV_DIR))
