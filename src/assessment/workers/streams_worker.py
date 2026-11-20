"""Streams worker (LYP/.venv-fusion): frozen feature construction + clinical/MRI specialists + CatBoost F/R.

Feature construction is analysis/temporal_eval/infer_streams.frame (the frozen feature_stats.json: step-4
median/mode imputation, z-scoring, CogIndex, volume ratios), unchanged. The model loop mirrors
infer_streams.streams / fusion_v3/test_streams.py, with the models loaded once instead of per call.
Inputs travel as CSV files (read with pandas exactly like the step-2 table); outputs as one .npz.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "analysis", "temporal_eval"))
import infer_streams as IS  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "analysis", "explainability"))
sys.path.insert(0, IS.FV3)
import mri_shap_lib as MS  # noqa: E402
import specialists as SP  # noqa: E402
from catboost import Pool  # noqa: E402

from src.assessment.workers.client import serve  # noqa: E402

FROZEN = IS.FROZEN
_M = {}


def load():
    if not _M:
        _M["stats"] = IS.FF.load_stats(os.path.join(FROZEN, "feature_stats.json"))
        _M["cols"] = json.load(open(os.path.join(FROZEN, "columns.json")))
        for s in IS.M.SEEDS:
            _M[f"clin_{s}"] = IS.cb(os.path.join(FROZEN, f"clinical_{s}.cbm"))
            _M[f"mri_{s}"] = pickle.load(open(os.path.join(FROZEN, f"mri_{s}.pkl"), "rb"))
            _M[f"cbF_{s}"] = IS.cb(os.path.join(FROZEN, f"catboost_F_{s}.cbm"))
            _M[f"cbR_{s}"] = IS.cb(os.path.join(FROZEN, f"catboost_R_{s}.cbm"))
    return _M


def streams(raw_csv, scan_csv, out_npz):
    """raw_csv: key + the 35 source features (no label). scan_csv: key, ALZ, unreliable, pred_age, age_at_scan."""
    m = load()
    raw = pd.read_csv(raw_csv, low_memory=False)
    raw.insert(1, "RID", np.arange(len(raw)))
    raw["label"] = -1  # frame() carries a label column; inference never uses it
    scan = pd.read_csv(scan_csv)
    scan["key"] = scan.key.astype(str)
    raw["key"] = raw.key.astype(str)
    d = IS.frame(raw, scan, m["stats"])
    cols = m["cols"]
    out = {"key": d.key.to_numpy().astype(str), "has": d.has_mri.to_numpy(), "ba_ok": d.ba_ok.to_numpy(),
           "extras": d[cols["extras"]].to_numpy(dtype=float), "cdrsb": d.CDRSB.to_numpy(dtype=float)}
    for s in IS.M.SEEDS:
        out[f"clin_{s}"] = m[f"clin_{s}"].predict_proba(d[cols["clinical_top12"][str(s)]])
        out[f"mri_{s}"] = m[f"mri_{s}"].predict_proba(d)
        out[f"cbF_{s}"] = m[f"cbF_{s}"].predict_proba(d[cols["catboost_F_cols"]])
        out[f"cbR_{s}"] = m[f"cbR_{s}"].predict_proba(d[cols["catboost_R_cols"]])
    out.update(explain(m, d))
    np.savez(out_npz, **out)
    return {"n": len(d), "npz": out_npz}


def mri_background(m):
    """z-scored MRI-specialist inputs of the train subjects with a scan (SHAP background). Local cache, never committed."""
    path = os.path.join(ROOT, "var", "cache", "mri_shap_bg.npy")
    if os.path.exists(path):
        return np.load(path)
    train = list(pd.read_csv(os.path.join(IS.FF.PROCESSED, "train_rids.csv")).RID)
    tr = IS.FF.build(train, m["stats"], "fit")
    zm, _ = MS.design(m["mri_42"], tr[tr.has_mri])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, zm)
    return zm


def explain(m, d):
    """Clinical SHAP (mean of the 5 seeds, mapped on the 29 clinical features) and exact linear MRI-specialist SHAP,
    as src/card/build_card_data.py computed them for the card."""
    feats = SP.clinical_features()
    cols = m["cols"]
    shap = np.zeros((len(IS.M.SEEDS), len(d), len(feats), 4))
    for si, s in enumerate(IS.M.SEEDS):
        top = cols["clinical_top12"][str(s)]
        sv = m[f"clin_{s}"].get_feature_importance(Pool(d[top], cat_features=[c for c in top if c == "PTGENDER"]),
                                                    type="ShapValues")
        for j, c in enumerate(top):
            shap[si, :, feats.index(c), :] = sv[:, :, j]
    has = d.has_mri.to_numpy()
    mri = m["mri_42"]
    phi = np.full((len(d), len(MS.FEATURES), 4), np.nan)
    imp = np.zeros((len(d), len(MS.FEATURES)), bool)
    if has.any():
        zm, miss = MS.design(mri, d[has])
        phi[has], _ = MS.shap_values(mri, zm, mri_background(m))
        imp[has] = miss
    gap = (d.pred_age - mri.beta) / mri.alpha - d.age_at_scan
    return {"cshap": shap.mean(0).astype(np.float32), "mshap_phi": phi.astype(np.float32), "mshap_imputed": imp,
            "clinical_features": np.array(feats), "mri_features": np.array(MS.FEATURES),
            "cogindex": d.CogIndex.to_numpy(float), "corrected_gap": gap.to_numpy(float),
            "ratios_z": d[IS.FF.RATIOS].to_numpy(float)}


if __name__ == "__main__":
    serve({"streams": streams, "ping": lambda: "streams"})
