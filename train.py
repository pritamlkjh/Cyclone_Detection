"""STEP 2 - Train on Kalpana.
   (a) 5-fold chronological CV (validate on one year at a time) -> honest estimate + learning curves
   (b) final model on ALL Kalpana years -> final.pt  (used later on INSAT)
Each training run = Stage 1 'head' (backbone frozen)  ->  Stage 2 'finetune' (layer3/4 unfrozen, lower LR)."""
import argparse, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from common import *
from dataset import CycloneSet
from model import build_model, set_stage, set_train_mode, param_groups
from splits import year_folds

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="resnet18"); ap.add_argument("--epochs_head", type=int, default=5)
ap.add_argument("--epochs_ft", type=int, default=15); ap.add_argument("--lr_head", type=float, default=1e-3)
ap.add_argument("--lr_ft", type=float, default=1e-4); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--skip_cv", action="store_true"); ap.add_argument("--workers", type=int, default=0)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(a.seed); np.random.seed(a.seed)

X = np.load(CACHE_DIR / "X.npy", mmap_mode="r")
meta = pd.read_csv(CACHE_DIR / "meta.csv", parse_dates=["time"]); y = meta.label.to_numpy()
norm = load_norm()

@torch.no_grad()
def predict(model, loader):
    model.eval(); P = []
    for xb, _ in loader: P.append(torch.sigmoid(model(xb.to(dev)).squeeze(1)).cpu())
    return torch.cat(P).numpy()

def metrics(yt, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(acc=accuracy_score(yt, yh), precision=precision_score(yt, yh, zero_division=0),
                recall=recall_score(yt, yh), f1=f1_score(yt, yh),
                auc=roc_auc_score(yt, p) if len(set(yt)) > 1 else float("nan"))

def run(train_idx, val_idx=None, tag=""):
    """Returns (trained model, history, val probabilities)."""
    ytr = y[train_idx]
    pos_w = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)              # handles 18% cyclone / 82% non-cyclone imbalance
    tl = DataLoader(CycloneSet(X[train_idx], ytr, norm, True), a.batch, shuffle=True, num_workers=a.workers, drop_last=True)
    vl = DataLoader(CycloneSet(X[val_idx], y[val_idx], norm), a.batch, num_workers=a.workers) if val_idx is not None else None
    model = build_model(a.arch).to(dev); hist = []
    for stage, ep, lr in [("head", a.epochs_head, a.lr_head), ("finetune", a.epochs_ft, a.lr_ft)]:
        set_stage(model, stage)
        opt = torch.optim.AdamW(param_groups(model, stage, lr), weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
        for e in range(ep):
            set_train_mode(model, stage); tot = 0
            for xb, yb in tl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); loss = lossf(model(xb).squeeze(1), yb); loss.backward(); opt.step(); tot += loss.item() * len(yb)
            sched.step(); rec = dict(stage=stage, epoch=e + 1, train_loss=tot / (len(tl) * a.batch))
            if vl is not None:
                p = predict(model, vl); rec.update({f"val_{k}": v for k, v in metrics(y[val_idx], p).items()})
            hist.append(rec); print(tag, rec, flush=True)
    return model, hist, (predict(model, vl) if vl is not None else None)

def main():
    OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
    if not a.skip_cv:
        rows, oof_y, oof_p = [], [], []
        for yr, tr, va in year_folds(meta):
            _, hist, p = run(tr, va, f"[val {yr}]")
            pd.DataFrame(hist).to_csv(OUT / f"history_{yr}.csv", index=False)
            rows.append(dict(val_year=yr, **metrics(y[va], p))); oof_y.append(y[va]); oof_p.append(p)
        cv = pd.DataFrame(rows); cv.to_csv(OUT / "cv_results.csv", index=False)
        print(cv.round(3).to_string(index=False)); print("MEAN:\n", cv.drop(columns="val_year").mean().round(3))
        oy, op = np.concatenate(oof_y), np.concatenate(oof_p)
        thr = max(np.linspace(0.1, 0.9, 81), key=lambda t: f1_score(oy, op >= t))   # threshold from out-of-fold predictions
    else:
        thr = 0.5

    print("Training FINAL model on all Kalpana years ...")
    model, hist, _ = run(np.arange(len(y)), None, "[final]")
    torch.save(dict(state=model.state_dict(), arch=a.arch, norm=norm, threshold=float(thr)), OUT / "final.pt")
    print("saved outputs/final.pt  threshold =", round(float(thr), 3))


if __name__ == "__main__":   # required on Windows (DataLoader workers re-import this file)
    main()
