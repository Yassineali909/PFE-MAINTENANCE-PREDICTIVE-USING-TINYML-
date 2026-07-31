import os, numpy as np, pandas as pd
from scipy.fft import fft
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATASET_PATH = "/mnt/hgfs/csv"
LABELS = ["ball", "inner_race", "outer_race", "normal"]

X_raw, y_raw = [], []
for label in LABELS:
    folder = os.path.join(DATASET_PATH, label)
    for fname in os.listdir(folder):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, fname))
            X_raw.append(df["vibration"].values)
            y_raw.append(label)

X_raw = np.array(X_raw); y_raw = np.array(y_raw)
print(f"Total windows: {len(X_raw)}")

def extract_features(w):
    rms = np.sqrt(np.mean(w**2)); peak = np.max(np.abs(w))
    crest = peak / (rms + 1e-9)
    spectrum = np.abs(fft(w))[:len(w)//2]
    top5 = np.sort(spectrum)[::-1][:5]
    return np.concatenate([[rms, peak, crest, np.mean(w), np.std(w),
                            pd.Series(w).skew(), pd.Series(w).kurtosis()], top5])

X = np.array([extract_features(w) for w in X_raw])
le = LabelEncoder(); y = le.fit_transform(y_raw)

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
sc = StandardScaler(); sc.fit(X_train)

NAMES = ["rms","peak","crest","mean","std","skew","kurt",
         "fft1","fft2","fft3","fft4","fft5"]

print("\n=== ORDRE DES CLASSES (index -> nom) ===")
for i, c in enumerate(le.classes_):
    print(f"  {i} -> {c}")

print("\n=== FEATURES : plage BRUTE (avant scaling) ===")
for i, n in enumerate(NAMES):
    print(f"  {n:6s} min={X[:,i].min():14.4f}  max={X[:,i].max():14.4f}  mean={X[:,i].mean():14.4f}")

print("\n=== PARAMS SCALER (a copier dans le firmware) ===")
print("const float scaler_mean[12]  = {" + ", ".join(f"{v:.6f}f" for v in sc.mean_) + "};")
print("const float scaler_scale[12] = {" + ", ".join(f"{v:.6f}f" for v in sc.scale_) + "};")
