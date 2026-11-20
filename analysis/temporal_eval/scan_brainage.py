"""DeepBrainNet brain age per scan (run with TF_USE_LEGACY_KERAS=1 in LYP/.venv-deepbrainnet). Inference only.

Mirrors analysis/fusion_v3/brainage_split.py: frozen weights (sha256 checked), no brain age if the
scan's m1 registration check corr_vs_mni < 0.15.
--gate      G1b brain-age part: original TEST baseline mni volumes vs brainage_test.csv pred_age.
--ingestion every usable scan of private/scans.csv (mni from dataset/mri_prep_patha) -> private/brainage.csv.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "analysis", "brain_age"))
import compute_brainage as CB  # noqa: E402
import dbn_smoke as D  # noqa: E402

PRIVATE = os.path.join(HERE, "private")


def run(model, paths):
    out = []
    for i, p in enumerate(paths, 1):
        out.append(D.predict(model, D.to_dbn_grid(p))[0])
        if i % 100 == 0:
            print(f"{i}/{len(paths)}", flush=True)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gate", action="store_true")
    g.add_argument("--ingestion", action="store_true")
    a = ap.parse_args()
    with open(os.path.join(D.DBN, "Models", "DBN_model.h5"), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == CB.SHA256
    model = D.load_model()
    if a.gate:
        ref = pd.read_csv(os.path.join(ROOT, "analysis", "fusion_v3", "brainage_test.csv")).dropna(subset=["pred_age"])
        pred = run(model, [os.path.join(ROOT, "dataset", "mri_prep", "mni", f"{p}.nii.gz") for p in ref.PTID])
        d = float(np.abs(pred - ref.pred_age.to_numpy()).max())
        print(f"G1b brain age: TEST n={len(ref)} max |pred_age - brainage_test| = {d:.5f} y (pass {d <= 0.001})")
        return
    s = pd.read_csv(os.path.join(PRIVATE, "scans.csv"), low_memory=False)
    s = s[s.usable].reset_index(drop=True)
    s["excluded"] = s.corr_vs_mni < 0.15
    todo = s[~s.excluded]
    pred = run(model, [os.path.join(ROOT, "dataset", "mri_prep_patha", "mni", f"{k}.nii.gz") for k in todo.prep_key])
    s["pred_age"] = np.nan
    s.loc[todo.index, "pred_age"] = pred
    s.rename(columns={"prep_key": "key"})[["key", "RID", "age_at_scan", "pred_age", "corr_vs_mni", "excluded"]] \
        .to_csv(os.path.join(PRIVATE, "brainage.csv"), index=False)
    print(f"brain age for {len(todo)} scans; {int(s.excluded.sum())} excluded (corr_vs_mni < 0.15)")


if __name__ == "__main__":
    main()
