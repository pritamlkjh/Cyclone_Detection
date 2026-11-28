"""STEP 3 - Test the Kalpana-trained model on INSAT images (never seen in training).

Easiest (folder name = label):
    python test_insat.py --cyclone_dir Insat_cyclone --non_cyclone_dir Insat_non_cyclone
Other options:
    python test_insat.py --insat_dir some_folder                      # predictions only, no scores
    python test_insat.py --insat_dir some_folder --labels labels.csv  # labels.csv: file,label
"""
import argparse, numpy as np, pandas as pd, torch
from common import *
from model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--cyclone_dir"); ap.add_argument("--non_cyclone_dir")
ap.add_argument("--insat_dir"); ap.add_argument("--labels")
ap.add_argument("--ckpt", default=str(ROOT / "outputs/final.pt"))
a = ap.parse_args()

items = []                                   # (path, label or None)
if a.cyclone_dir:     items += [(p, 1) for p in sorted(Path(a.cyclone_dir).glob("*.npz"))]
if a.non_cyclone_dir: items += [(p, 0) for p in sorted(Path(a.non_cyclone_dir).glob("*.npz"))]
if a.insat_dir:       items += [(p, None) for p in sorted(Path(a.insat_dir).glob("*.npz"))]
if not items: raise SystemExit("No .npz files found - check the folder arguments.")

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); norm = ck["norm"]
model = build_model(ck["arch"], pretrained=False); model.load_state_dict(ck["state"]); model.eval()

rows = []
for p, lab in items:
    tir, wv = load_raw(p)
    x = normalise(make_3ch(resize_pair(tir, wv)), norm)           # identical pre-processing to Kalpana
    with torch.no_grad(): prob = torch.sigmoid(model(torch.from_numpy(x)[None])).item()
    rows.append(dict(file=p.name, label=lab, prob_cyclone=round(prob, 4), pred=int(prob >= ck["threshold"]), threshold=round(ck["threshold"], 4)))
df = pd.DataFrame(rows)
if a.labels:
    df = df.drop(columns="label").merge(pd.read_csv(a.labels), on="file", how="left")
(ROOT / "outputs").mkdir(exist_ok=True)
df.to_csv(ROOT / "outputs/insat_predictions.csv", index=False)
print(f"{len(df)} INSAT images | decision threshold used: {ck['threshold']:.2f}")

lab = df.dropna(subset=["label"]).astype({"label": int})
if len(lab):
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    print(classification_report(lab.label, lab.pred, target_names=["non-cyclone", "cyclone"], zero_division=0))
    print("confusion matrix  rows=true, cols=predicted  [[TN FP],[FN TP]]:\n", confusion_matrix(lab.label, lab.pred, labels=[0, 1]))
    if lab.label.nunique() > 1: print("AUC:", round(roc_auc_score(lab.label, lab.prob_cyclone), 3))
else:
    print(df.to_string(index=False))
print("saved outputs/insat_predictions.csv")
