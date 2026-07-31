import numpy as np, pandas as pd, glob, os
for c in ["ball","inner_race","normal","outer_race"]:
    f = sorted(glob.glob(f"/mnt/hgfs/csv/{c}/*.csv"))[0]
    w = pd.read_csv(f)["vibration"].values.astype(np.float32)
    print("%-12s rms=%.4f peak=%.4f kurt=%+.2f std=%.4f" % (
        c, np.sqrt(np.mean(w**2)), np.max(np.abs(w)),
        pd.Series(w).kurtosis(), np.std(w)))
