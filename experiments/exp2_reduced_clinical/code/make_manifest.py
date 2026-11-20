"""Write manifest.yaml (LYP/.venv-fusion + PYTHONPATH=LYP/.pylib-exp2): per tier the exact
columns, n features, train missingness per column, subjects kept (imputed, none dropped),
top-K, KAN input width and F-CB width."""
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

data = C.load()
(tr, rtr), (va, rva) = data["train"], data["val"]
src = {"Full-paper": "analysis/kan_fusion/fusion_trainval.csv (paper step-4 values, global train fit)"}
out = {"generated_by": "code/make_manifest.py", "preregistration": "PREREGISTRATION.md (commit de368a1)",
       "excluded_from_reduced_tiers": sorted(C.EXCLUDED), "tiers": {}}
for t, cols in C.TIERS.items():
    miss = {}
    for c in cols:
        if c == "CogIndex":
            miss[c] = round(float(rtr[["MMSE", "CDRSB", "ADAS13"]].isna().any(axis=1).mean()), 4) if t != "Full-paper" else 0.0
        else:
            miss[c] = round(float((tr if t == "Full-paper" else rtr)[c].isna().mean()), 4)
    k = min(12, len(cols))
    out["tiers"][t] = {
        "columns": list(cols), "n_features": len(cols), "top_k": k, "selection": "none" if t == "T0" else "in-fold mean |SHAP|",
        "source": src.get(t, "trainval_extra.parquet (raw step-1 values; in-fold step-4 rule)"),
        "train_missingness": miss, "max_train_missingness": max(miss.values()),
        "subjects": {"train": len(tr), "val": len(va)},
        "kan_extras": ("AGE, SEX, PTEDUCAT" if t == "T0" else "AGE, CDRSB, CogIndex, MMSE (fixed)" if t == "Full-paper"
                       else "AGE + in-fold SHAP top-3 non-AGE"),
        "kan_input_width": 12 if t == "T0" else 13,
        "fcb_width": f"{k} (+1 if AGE not in top-K) + 9 MRI inputs",
        "in_trend_R3": t in C.TREND,
    }
out["mri_specialist"] = {"inputs": C.RATIOS + ["ALZ", "BAG", "AGE", "SEX"], "model": "logreg C=0.1 (frozen)",
                         "subjects_with_scan": {"train": int(tr.has_mri.sum()), "val": int(va.has_mri.sum())}}
out["class_counts"] = {"train": tr.label.value_counts().sort_index().rename(dict(enumerate(C.NAMES))).to_dict(),
                       "val": va.label.value_counts().sort_index().rename(dict(enumerate(C.NAMES))).to_dict()}
with open(os.path.join(C.EXP, "manifest.yaml"), "w") as f:
    yaml.safe_dump(out, f, sort_keys=False, width=140)
for t, v in out["tiers"].items():
    print(f"{t:11s} n={v['n_features']:2d} K={v['top_k']:2d} max_miss={v['max_train_missingness']:.4f} KAN width {v['kan_input_width']}")
print(out["class_counts"], out["mri_specialist"]["subjects_with_scan"])
