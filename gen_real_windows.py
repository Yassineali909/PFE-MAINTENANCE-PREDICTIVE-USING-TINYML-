import numpy as np, pandas as pd, os, glob

CLASSES = ["ball", "inner_race", "normal", "outer_race"]
CSV_ROOT = "/mnt/hgfs/csv"
WINDOWS_PER_CLASS = 2
WINDOW_SIZE = 1024

def pick_windows(class_dir, n):
    files = sorted(glob.glob(os.path.join(class_dir, "*.csv")))
    if not files:
        raise FileNotFoundError("Aucun CSV dans " + class_dir)
    idx = np.linspace(0, len(files) - 1, n).astype(int)
    chosen = []
    for i in idx:
        w = pd.read_csv(files[i])["vibration"].values[:WINDOW_SIZE].astype(np.float32)
        if len(w) < WINDOW_SIZE:
            continue
        k = pd.Series(w).kurtosis()
        print("  %-30s kurtosis=%+.2f" % (os.path.basename(files[i]), k))
        chosen.append(w)
    return chosen

lines = ["#pragma once", "#include <Arduino.h>",
         "#define WINDOW_SIZE %d" % WINDOW_SIZE,
         "#define WINDOWS_PER_CLASS %d" % WINDOWS_PER_CLASS,
         "#define NUM_SIM_CLASSES %d" % len(CLASSES), ""]

print("Kurtosis (doit etre >> 0 pour les defauts, ~0 pour normal):")
all_data = []
for c in CLASSES:
    print("[%s]" % c)
    all_data.append(pick_windows(os.path.join(CSV_ROOT, c), WINDOWS_PER_CLASS))

lines.append("static const float CWRU_WINDOWS[%d][%d][%d] PROGMEM = {" % (len(CLASSES), WINDOWS_PER_CLASS, WINDOW_SIZE))
for ci, ws in enumerate(all_data):
    lines.append("  { // %s" % CLASSES[ci])
    for w in ws:
        lines.append("    {" + ",".join("%.6ff" % x for x in w) + "},")
    lines.append("  },")
lines.append("};")
lines.append("")
lines.append("inline void generate_cwru_window(float* buffer, int fault_class) {")
lines.append("    static int win_idx[NUM_SIM_CLASSES] = {0};")
lines.append("    int w = win_idx[fault_class];")
lines.append("    for (int i = 0; i < WINDOW_SIZE; i++)")
lines.append("        buffer[i] = pgm_read_float(&CWRU_WINDOWS[fault_class][w][i]);")
lines.append("    win_idx[fault_class] = (w + 1) % WINDOWS_PER_CLASS;")
lines.append("}")
lines.append("")

with open("esp32_s3_project/src/cwru_real_windows.h", "w") as f:
    f.write("\n".join(lines))
print("\nEcrit : esp32_s3_project/src/cwru_real_windows.h")
