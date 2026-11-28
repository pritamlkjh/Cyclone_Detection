"""Shared helpers: file-name parsing, loading .npz images, resizing, normalising.
The SAME functions are used for Kalpana (train) and INSAT (test), so the only
difference between the two is the satellite itself - exactly what transfer learning should face."""
import re, json
from datetime import datetime
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT if (ROOT / "Kalpana_Cyclones").exists() else ROOT / "data"  # supports data/ or project root
CACHE_DIR = ROOT / "cache"
IMG = 224                         # ResNet's native input size
_PAT = re.compile(r"_(\d{2})([A-Za-z]{3})(\d{4})_(\d{4})_")

def parse_time(name: str) -> datetime:
    """K1VHR_01NOV2010_0300_...  or  3DIMG_01DEC2015_1500_...  ->  datetime"""
    d, mon, y, hm = _PAT.search(name).groups()
    return datetime.strptime(f"{d}{mon.upper()}{y}{hm}", "%d%b%Y%H%M")

def load_raw(path):
    z = np.load(path, allow_pickle=True)
    tir = np.nan_to_num(z["tir"].astype(np.float32))
    wv = np.nan_to_num(z["wv"].astype(np.float32))
    return tir, wv

def resize_pair(tir, wv, size=IMG):
    """500x500 -> size x size (INTER_AREA = proper averaging when shrinking). Returns (2,H,W)."""
    r = lambda a: cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
    return np.stack([r(tir), r(wv)])

def make_3ch(x2):
    """(2,H,W) -> (3,H,W): [TIR, WV, TIR-WV]. 3 channels so ImageNet ResNet weights fit without surgery.
    TIR-WV highlights deep convective cloud tops (typical of cyclones)."""
    return np.concatenate([x2, (x2[0] - x2[1])[None]], 0)

def load_norm():
    return json.load(open(CACHE_DIR / "norm.json"))

def normalise(x3, norm):
    m = np.array(norm["mean"], np.float32)[:, None, None]
    s = np.array(norm["std"], np.float32)[:, None, None]
    return (x3 - m) / s
