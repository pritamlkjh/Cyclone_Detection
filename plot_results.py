"""Graphs for the report: training curves, cross-validation results, INSAT test (confusion matrix,
accuracy / precision / recall / F1 / AUC, ROC, precision-recall, probability histogram).

    python plot_results.py              # saves PNGs into outputs/plots/
    python plot_results.py --show       # also opens the figures in windows

Reads (all created by train.py / test_insat.py / the GUI):
    outputs/history_<year>.csv, outputs/cv_results.csv, outputs/insat_predictions.csv
Anything missing is skipped, so it works at every stage of the project."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)

ROOT = Path(__file__).resolve().parent
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]


def compute_metrics(y, pred, prob):
    y, pred, prob = np.asarray(y).astype(int), np.asarray(pred).astype(int), np.asarray(prob, float)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: a / b if b else 0.0
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    return dict(accuracy=float(div(tp + tn, len(y))), precision=float(p), recall=float(r), specificity=float(div(tn, tn + fp)),
                f1=float(div(2 * p * r, p + r)), auc=float(roc_auc_score(y, prob)) if len(set(y)) > 1 else float("nan"),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=len(y))


def _insat(out):
    f = out / "insat_predictions.csv"
    if not f.exists(): return None
    df = pd.read_csv(f)
    if "label" not in df: return None
    df = df.dropna(subset=["label"]).astype({"label": int})
    return df if len(df) else None


def fig_training(out, make):
    hs = {p.stem.split("_")[1]: pd.read_csv(p) for p in sorted(out.glob("history_*.csv"))}
    if not hs: return None
    keys = [("train_loss", "Training loss"), ("val_accuracy", "Validation accuracy"), ("val_precision", "Validation precision"),
            ("val_recall", "Validation recall"), ("val_f1", "Validation F1"), ("val_auc", "Validation AUC")]
    fig = make(figsize=(14, 7.5))
    first = next(iter(hs.values())); ft = first.index[first.stage == "finetune"]
    for i, (k, title) in enumerate(keys):
        ax = fig.add_subplot(2, 3, i + 1)
        k_use = k if k in first else k.replace("accuracy", "acc")      # train.py stores accuracy as 'acc'
        for yr, h in hs.items():
            if k_use in h: ax.plot(range(1, len(h) + 1), h[k_use], marker=".", lw=1.3, label=f"validate on {yr}")
        if len(ft):
            ax.axvline(ft[0] + 0.5, color="k", ls="--", lw=1)
            if i == 0: ax.text(ft[0] + 0.7, ax.get_ylim()[1] * 0.9, "fine-tuning starts", fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_xlabel("epoch  (head training → fine-tuning)", fontsize=8); ax.grid(alpha=.3)
        if k != "train_loss": ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.axes[0].legend(fontsize=7)
    fig.suptitle("Training on Kalpana: chronological cross-validation (one whole year held out per fold)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95)); return fig


def _cv_table(out):
    f = out / "cv_results.csv"
    if not f.exists(): return None
    cv = pd.read_csv(f).rename(columns={"acc": "accuracy"}); return cv


def fig_cv(out, make):
    cv = _cv_table(out)
    if cv is None: return None
    fig = make(figsize=(11, 5)); ax = fig.add_subplot(111)
    n = len(cv) + 1; w = .8 / n; x = np.arange(len(METRICS))
    for i, (_, row) in enumerate(cv.iterrows()):
        ax.bar(x + i * w, [row[m] for m in METRICS], w, label=f"val {int(row.val_year)}")
    mean = [cv[m].mean() for m in METRICS]
    b = ax.bar(x + len(cv) * w, mean, w, color="k", label="mean")
    for r, v in zip(b, mean): ax.text(r.get_x() + w / 2, v + .01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x + .4 - w / 2); ax.set_xticklabels([m.upper() if m == "auc" else m.capitalize() for m in METRICS])
    ax.set_ylim(0, 1.1); ax.grid(axis="y", alpha=.3); ax.legend(ncol=n, fontsize=8, loc="lower center")
    ax.set_title("Kalpana cross-validation results (each bar = a held-out year)"); fig.tight_layout(); return fig


def fig_insat_matrix(out, make):
    df = _insat(out)
    if df is None: return None
    m = compute_metrics(df.label, df.pred, df.prob_cyclone)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig = make(figsize=(12, 5)); a1, a2 = fig.add_subplot(121), fig.add_subplot(122)
    pct = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    a1.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a1.text(j, i, f"{cm[i, j]}\n({pct[i, j]*100:.1f}%)", ha="center", va="center", fontsize=13,
                    color="white" if pct[i, j] > .5 else "black")
    a1.set_xticks([0, 1]); a1.set_xticklabels(["Non-cyclone", "Cyclone"]); a1.set_yticks([0, 1]); a1.set_yticklabels(["Non-cyclone", "Cyclone"])
    a1.set_xlabel("Predicted"); a1.set_ylabel("True"); a1.set_title("Confusion matrix (INSAT test)")
    names = ["accuracy", "precision", "recall", "specificity", "f1", "auc"]
    vals = [m[k] for k in names]
    bars = a2.bar([n.upper() if n == "auc" else n.capitalize() for n in names], np.nan_to_num(vals), color="#4c78a8")
    for r, v in zip(bars, vals): a2.text(r.get_x() + r.get_width() / 2, (0 if np.isnan(v) else v) + .01, "n/a" if np.isnan(v) else f"{v:.3f}", ha="center", fontsize=9)
    a2.set_ylim(0, 1.1); a2.grid(axis="y", alpha=.3); a2.set_title(f"Metrics on {m['n']} INSAT images")
    fig.tight_layout(); return fig


def fig_insat_curves(out, make):
    df = _insat(out)
    if df is None: return None
    thr = float(df.threshold.iloc[0]) if "threshold" in df else 0.5
    fig = make(figsize=(14, 4.8)); a1, a2, a3 = fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)
    if df.label.nunique() > 1:
        fpr, tpr, _ = roc_curve(df.label, df.prob_cyclone)
        a1.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(df.label, df.prob_cyclone):.3f}"); a1.plot([0, 1], [0, 1], "k--", lw=1)
        pr, rc, _ = precision_recall_curve(df.label, df.prob_cyclone)
        a2.plot(rc, pr, lw=2, label=f"AP = {average_precision_score(df.label, df.prob_cyclone):.3f}")
        a1.legend(); a2.legend()
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate"); a1.set_title("ROC curve"); a1.grid(alpha=.3)
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision"); a2.set_title("Precision-recall curve"); a2.grid(alpha=.3)
    bins = np.linspace(0, 1, 21)
    a3.hist(df.prob_cyclone[df.label == 0], bins, alpha=.7, label="true non-cyclone", color="#2e9d5b")
    a3.hist(df.prob_cyclone[df.label == 1], bins, alpha=.7, label="true cyclone", color="#c0392b")
    a3.axvline(thr, color="k", ls="--", label=f"threshold {thr:.2f}")
    a3.set_xlabel("Predicted cyclone probability"); a3.set_ylabel("Number of images"); a3.set_title("How confident is the model?"); a3.legend(fontsize=8)
    fig.tight_layout(); return fig


def fig_compare(out, make):
    cv, df = _cv_table(out), _insat(out)
    if cv is None or df is None: return None
    m = compute_metrics(df.label, df.pred, df.prob_cyclone)
    fig = make(figsize=(9, 5)); ax = fig.add_subplot(111); x = np.arange(len(METRICS)); w = .38
    ax.bar(x - w / 2, [cv[k].mean() for k in METRICS], w, yerr=[cv[k].std() for k in METRICS], capsize=3, label="Kalpana CV (mean ± std over years)")
    ax.bar(x + w / 2, np.nan_to_num([m[k] for k in METRICS]), w, label="INSAT test")
    ax.set_xticks(x); ax.set_xticklabels([k.upper() if k == "auc" else k.capitalize() for k in METRICS])
    ax.set_ylim(0, 1.1); ax.grid(axis="y", alpha=.3); ax.legend(loc="lower center")
    ax.set_title("Effect of changing satellite: Kalpana validation vs INSAT test"); fig.tight_layout(); return fig


def build_figures(out=None, make=None):
    """-> {tab name: Figure}. Used by this script and by the GUI's 'Show graphs' button."""
    out = Path(out or ROOT / "outputs"); make = make or (lambda **kw: Figure(**kw))
    items = [("Training curves", fig_training), ("Kalpana CV", fig_cv), ("INSAT matrix + metrics", fig_insat_matrix),
             ("INSAT ROC / PR / probs", fig_insat_curves), ("Kalpana vs INSAT", fig_compare)]
    figs = {}
    for name, fn in items:
        f = fn(out, make)
        if f is not None: figs[name] = f
    return figs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--outputs", default=str(ROOT / "outputs")); ap.add_argument("--show", action="store_true")
    a = ap.parse_args(); out = Path(a.outputs)
    make = None
    if a.show:
        import matplotlib.pyplot as plt; make = plt.figure
    figs = build_figures(out, make)
    if not figs: raise SystemExit(f"Nothing to plot in {out} - run train.py / test_insat.py first.")
    (out / "plots").mkdir(exist_ok=True)
    for name, fig in figs.items():
        p = out / "plots" / (name.lower().replace(" + ", "_").replace(" / ", "_").replace(" ", "_") + ".png")
        fig.savefig(p, dpi=150); print("saved", p)
    df = _insat(out)
    if df is not None:
        m = compute_metrics(df.label, df.pred, df.prob_cyclone); print("\nINSAT test:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    if a.show: plt.show()
