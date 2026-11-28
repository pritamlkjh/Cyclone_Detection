"""Chronological (year-block) cross-validation.
Every image of a given year lives in the SAME chunk. Fold k = validate on year k, train on all other years.
Why years and not random k-fold?  Frames are 3 h apart, so neighbouring frames look almost identical.
A random split would put near-duplicates in train AND validation -> fake high scores (leakage)."""
import pandas as pd
from common import CACHE_DIR

def year_folds(meta: pd.DataFrame):
    """Yields (val_year, train_idx, val_idx). Years with no cyclone images (2008) cannot be a
    meaningful validation fold, so they are always kept in training."""
    cyc_years = sorted(meta.loc[meta.label == 1, "year"].unique())
    for y in cyc_years:
        yield y, meta.index[meta.year != y].to_numpy(), meta.index[meta.year == y].to_numpy()

if __name__ == "__main__":
    meta = pd.read_csv(CACHE_DIR / "meta.csv", parse_dates=["time"])
    for y, tr, va in year_folds(meta):
        m = meta.loc[va]
        assert set(meta.loc[tr, "year"]).isdisjoint(set(m.year))          # no year appears on both sides
        print(f"Fold val={y}: train {len(tr):4d} (cyc {int(meta.loc[tr,'label'].sum()):3d}) | "
              f"val {len(va):4d} (cyc {int(m.label.sum()):3d}) | val range {m.time.min():%d-%b-%Y} -> {m.time.max():%d-%b-%Y}")
