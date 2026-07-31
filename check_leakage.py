import glob, os
CSV="/mnt/hgfs/csv"; CLASSES=["ball","inner_race","normal","outer_race"]
for c in CLASSES:
    files = glob.glob(os.path.join(CSV,c,"*.csv"))
    sources = set(os.path.basename(f).split("_w")[0] for f in files)
    print("%-12s : %d fenetres, %d sources -> %s" % (c, len(files), len(sources), sorted(sources)))
