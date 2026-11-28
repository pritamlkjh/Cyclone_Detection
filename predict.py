"""Small helper used by the GUI (and usable on its own): load final.pt and predict one image pair."""
import numpy as np, torch
from common import ROOT, load_raw, resize_pair, make_3ch, normalise
from model import build_model

def load_npz_pair(path):
    """-> (tir, wv), each 500x500 float32 (raw counts)."""
    return load_raw(path)

def load_model_and_stats(ckpt=None):
    ck = torch.load(ckpt or ROOT / "outputs/final.pt", map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], pretrained=False)
    model.load_state_dict(ck["state"]); model.eval()
    return model, dict(norm=ck["norm"], threshold=float(ck["threshold"]))

def predict_from_arrays(tir, wv, model, stats):
    """Same pre-processing as training. Returns (cyclone probability, 'CYCLONE' / 'NO CYCLONE')."""
    x = normalise(make_3ch(resize_pair(tir, wv)), stats["norm"])
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(x)[None])).item()
    return prob, ("CYCLONE" if prob >= stats["threshold"] else "NO CYCLONE")
