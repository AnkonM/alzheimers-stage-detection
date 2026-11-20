"""Reference tables for the report (run with LYP/.venv-fusion from the repo root).

Population: the 569 clinical-train subjects only (decision D3). Val and test are never read.
Aggregates only are written (quantiles, counts); no subject-level row, ID or label list is written.

  train range       min and max of each clinical feature over the 569 (soft range: outside it the report warns of extrapolation);
  per-stage ranges  p25/p50/p75 of each clinical feature and each MRI-specialist input, by true stage;
  CN volume grid    101 quantiles (0..100) of each volume/ICV ratio over the CN-train subjects, so the report can
                    place a new ratio as a percentile. "Not age/sex-adjusted": the pooled CN-train distribution.

    python analysis/inference_refs/build_refs.py
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
V3 = os.path.join(ROOT, "analysis", "fusion_v3")
sys.path.insert(0, V3)
sys.path.insert(0, os.path.join(ROOT, "src"))
import final_features as FF  # noqa: E402
import specialists as S  # noqa: E402

NAMES = ["CN", "EMCI", "LMCI", "AD"]
OUT = os.path.join(HERE, "results", "reference_ranges.json")


def q3(x):
    x = x[~np.isnan(x)]
    return None if len(x) == 0 else {"n": int(len(x)), **dict(zip(("p25", "p50", "p75"), np.percentile(x, [25, 50, 75]).tolist()))}


def main():
    train = list(pd.read_csv(os.path.join(FF.PROCESSED, "train_rids.csv")).RID)
    assert len(train) == 569
    test = set(pd.read_csv(os.path.join(FF.PROCESSED, "test_rids.csv")).RID)
    assert not set(train) & test
    stats = FF.load_stats(os.path.join(V3, "frozen", "feature_stats.json"))
    d = FF.build(train, stats, "fit")
    raw = FF.raw_rows(train).set_index("RID").loc[d.RID].reset_index()
    mri = pickle.load(open(os.path.join(V3, "frozen", "mri_42.pkl"), "rb"))
    gap = ((d.pred_age - mri.beta) / mri.alpha - d.age_at_scan).where(d.ba_ok)
    vals = {}
    for c in S.clinical_features():
        vals[c] = (d[c] if c in ("CogIndex",) else raw[c]).to_numpy(float) if c != "PTGENDER" else None
    mri_vals = {r: (raw[v] / raw["ICV"]).to_numpy(float) for v, r in zip(FF.VOLUMES, FF.RATIOS)}
    mri_vals.update({"ALZ": d.ALZ.to_numpy(float), "BAG": gap.to_numpy(float), "AGE": raw["AGE"].to_numpy(float)})
    lab = d.label.to_numpy()

    def by_stage(x):
        return {NAMES[k]: q3(x[lab == k]) for k in range(4)}
    def span(x):
        x = x[~np.isnan(x)]
        return [float(x.min()), float(x.max())] if len(x) else None
    out = {"train_range": {c: span(v) for c, v in vals.items() if v is not None},
           "population": {"split": "train", "n": int(len(d)), "n_by_stage": {NAMES[k]: int((lab == k).sum()) for k in range(4)},
                          "note": "clinical-train subjects only (decision D3); aggregates only"},
           "clinical": {c: by_stage(v) for c, v in vals.items() if v is not None},
           "mri": {c: by_stage(v) for c, v in mri_vals.items()},
           "cn_volume_quantiles": {}}
    cn = lab == 0
    for r in FF.RATIOS:
        x = mri_vals[r][cn]
        x = x[~np.isnan(x)]
        out["cn_volume_quantiles"][r] = {"n": int(len(x)), "basis": "pooled CN-train, not age/sex-adjusted",
                                         "q": np.percentile(x, np.arange(101)).tolist()}
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT, out["population"])


if __name__ == "__main__":
    main()
