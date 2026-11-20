"""Frozen stream scores per scan (run with LYP/.venv-fusion). Inference only.

Mirrors analysis/fusion_v3/final_features.build + test_streams.py, keyed by scan
(`key`) instead of subject: the frozen feature_stats.json (fitted on the 691) is
applied to each scan's raw record, then the frozen clinical specialist, MRI
specialist and CatBoost F / R of every seed. Nothing is fitted.

--gate      G1a: stored TEST baseline inputs (step2 rows, scan_inputs_test.csv,
            brainage_test.csv) -> private/gate_test_streams.npz, compared with
            frozen_run/test_streams.npz.
--ingestion private/scans.csv + private/scan_inputs.csv -> private/ingestion_streams.npz.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FV3 = os.path.join(ROOT, "analysis", "fusion_v3")
sys.path.insert(0, FV3)
sys.path.insert(0, os.path.join(ROOT, "src"))
import final_features as FF  # noqa: E402
import metrics as M  # noqa: E402
import step4_processing as S4  # noqa: E402

FROZEN = os.path.join(FV3, "frozen")
PRIVATE = os.path.join(HERE, "private")
FEATURES = None  # the 35 step-2 source features, from feature_stats.json


def frame(raw, scan, stats):
    """FF.build on given frames. raw: key, RID, label + the 35 source features; scan: key, ALZ, unreliable,
    pred_age, age_at_scan."""
    imp = S4.apply_imputation(raw, stats["median"], stats["mode"])
    df = S4.add_cogindex(S4.apply_normalization(imp, stats["mean"], stats["std"]))
    for v, n in zip(FF.VOLUMES, FF.RATIOS):
        r = (raw[v] / raw["ICV"]).fillna(stats["ratio_median"][n])
        df[n] = (r - stats["ratio_mean"][n]) / stats["ratio_std"][n]
    df = df.merge(scan[["key", "ALZ", "unreliable", "pred_age", "age_at_scan"]], on="key", how="left")
    df["has_mri"] = df.ALZ.notna()
    df["ba_ok"] = df.pred_age.notna() & (df.unreliable != True)              # noqa: E712
    df["SEX"] = (df.PTGENDER == "Male").astype(float)
    assert len(df) == len(raw) and df.key.is_unique
    return df


def cb(path):
    m = CatBoostClassifier()
    m.load_model(path)
    return m


def streams(d):
    cols = json.load(open(os.path.join(FROZEN, "columns.json")))
    out = {"key": d.key.to_numpy().astype(str), "rids": d.RID.to_numpy(), "y": d.label.to_numpy(),
           "has": d.has_mri.to_numpy(), "ba_ok": d.ba_ok.to_numpy(),
           "extras": d[cols["extras"]].to_numpy(dtype=float), "cdrsb": d.CDRSB.to_numpy(dtype=float)}
    for s in M.SEEDS:
        out[f"clin_{s}"] = cb(os.path.join(FROZEN, f"clinical_{s}.cbm")).predict_proba(d[cols["clinical_top12"][str(s)]])
        out[f"mri_{s}"] = pickle.load(open(os.path.join(FROZEN, f"mri_{s}.pkl"), "rb")).predict_proba(d)
        out[f"cbF_{s}"] = cb(os.path.join(FROZEN, f"catboost_F_{s}.cbm")).predict_proba(d[cols["catboost_F_cols"]])
        out[f"cbR_{s}"] = cb(os.path.join(FROZEN, f"catboost_R_{s}.cbm")).predict_proba(d[cols["catboost_R_cols"]])
    return out


def gate_inputs(stats):
    rids = list(pd.read_csv(os.path.join(ROOT, "dataset", "processed", "test_rids.csv")).RID)
    raw = FF.raw_rows(rids)
    raw.insert(0, "key", raw.RID.astype(str))
    sc = FF.scan_tables(rids, "test")
    sc.insert(0, "key", sc.RID.astype(str))
    return raw, sc


def ingestion_inputs(stats):
    s = pd.read_csv(os.path.join(PRIVATE, "scans.csv"), low_memory=False)
    s = s[s.usable].copy()
    s["key"] = s.prep_key
    feats = stats["num"] + stats["cat"]
    raw = s[["key", "RID"] + feats].copy()
    raw["label"] = s.y_90.fillna(-1).astype(int)
    si = pd.read_csv(os.path.join(PRIVATE, "scan_inputs.csv"))
    ba = pd.read_csv(os.path.join(PRIVATE, "brainage.csv"))
    sc = s[["key", "age_at_scan"]].merge(si[["key", "ALZ", "unreliable"]], on="key", how="left") \
        .merge(ba[["key", "pred_age"]], on="key", how="left")
    return raw, sc


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gate", action="store_true")
    g.add_argument("--ingestion", action="store_true")
    a = ap.parse_args()
    stats = FF.load_stats(os.path.join(FROZEN, "feature_stats.json"))
    raw, sc = (gate_inputs if a.gate else ingestion_inputs)(stats)
    out = streams(frame(raw, sc, stats))
    os.makedirs(PRIVATE, exist_ok=True)
    name = "gate_test_streams.npz" if a.gate else "ingestion_streams.npz"
    np.savez(os.path.join(PRIVATE, name), **out)
    print(f"{name}: {len(out['key'])} rows")
    if a.gate:
        ref = np.load(os.path.join(FV3, "frozen_run", "test_streams.npz"))
        assert (ref["rids"] == out["rids"]).all()
        worst = {k: float(np.nanmax(np.abs(ref[k].astype(float) - out[k].astype(float)))) if ref[k].dtype != bool
                 else float((ref[k] != out[k]).sum()) for k in ref.files}
        bad = {k: v for k, v in worst.items() if not v <= 1e-6}
        print("G1a streams: max |diff| over all keys", max(worst.values()), "| keys above 1e-6:", bad or "none")


if __name__ == "__main__":
    main()
