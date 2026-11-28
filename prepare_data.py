"""STEP 1 - read every Kalpana .npz once, resize to 224x224, cache as one array + a metadata table."""
import numpy as np, pandas as pd, json
from common import *

rows, imgs = [], []
for folder, label in [("Kalpana_Cyclones", 1), ("Kalpana_Non_Cyclones", 0)]:
    for p in sorted((DATA_DIR / folder).glob("*.npz")):
        tir, wv = load_raw(p)
        imgs.append(resize_pair(tir, wv).astype(np.float16))   # counts <=1023 are exact in float16
        t = parse_time(p.name)
        rows.append(dict(file=p.name, label=label, time=t, year=t.year))
X = np.stack(imgs); del imgs
meta = pd.DataFrame(rows)
CACHE_DIR.mkdir(exist_ok=True)
np.save(CACHE_DIR / "X.npy", X)
meta.to_csv(CACHE_DIR / "meta.csv", index=False)

# per-channel mean/std of the 3-channel input (computed on a subsample of Kalpana only)
idx = np.random.RandomState(0).choice(len(X), min(400, len(X)), replace=False)
s = np.stack([make_3ch(X[i].astype(np.float32)) for i in idx])
json.dump(dict(mean=s.mean((0, 2, 3)).tolist(), std=s.std((0, 2, 3)).tolist()), open(CACHE_DIR / "norm.json", "w"))
print("cached", X.shape, "| cyclone:", int(meta.label.sum()), "non-cyclone:", int((1 - meta.label).sum()))
print(pd.crosstab(meta.year, meta.label))
print("norm:", open(CACHE_DIR / "norm.json").read())
