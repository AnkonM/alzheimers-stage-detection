"""Agreement of SynthSeg vs FreeSurfer 5.1 on the six model-input ratios (PLAN.md). Aggregate output only.

    python -m analysis.volume_agreement.analyse     (LYP/.venv-app, repo root)
"""
import glob
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
STRUCT = ["Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp"]
CORTICAL = ["Entorhinal", "Fusiform", "MidTemp"]
CORTICAL_FLOOR = 0.50  # declared in PLAN.md


def icc21(x, y):
    """ICC(2,1): two-way random effects, absolute agreement, single measurement (Shrout & Fleiss)."""
    d = np.column_stack([x, y])
    n, k = d.shape
    gm = d.mean()
    msr = k * ((d.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((d.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def boot_ci(x, y, n=2000):
    rng = np.random.default_rng(0)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        v.append(icc21(x[i], y[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def ba(ss, fs):
    diff = ss - fs
    bias, sd = diff.mean(), diff.std(ddof=1)
    m = fs.mean()
    return {"bias_pct_of_fs_mean": float(100 * bias / m), "loa_low_pct": float(100 * (bias - 1.96 * sd) / m),
            "loa_high_pct": float(100 * (bias + 1.96 * sd) / m)}


def main():
    rs = [json.load(open(p)) for p in sorted(glob.glob(str(HERE / "private" / "*" / "result.json")))]
    ok = [r for r in rs if "volumes" in r]
    fails = [r for r in rs if "volumes" not in r]
    cols = {}
    for s in STRUCT:
        cols[f"{s} / ICV"] = (np.array([r["volumes"][s] / r["volumes"]["ICV"] for r in ok]),
                              np.array([r["fs"][s] / r["fs"]["ICV"] for r in ok]))
    for s in STRUCT + ["ICV"]:
        cols[f"{s} (raw mm3)"] = (np.array([r["volumes"][s] for r in ok]), np.array([r["fs"][s] for r in ok]))
    res = {}
    for name, (ss, fs) in cols.items():
        res[name] = {"icc21": icc21(ss, fs), "icc21_ci95": boot_ci(ss, fs), **ba(ss, fs),
                     "pearson_r": float(np.corrcoef(ss, fs)[0, 1])}
    meta = {"n_requested": 25, "n_ok": len(ok), "n_failed": len(fails),
            "qc_min_range": [min(r["qc_min"] for r in ok), max(r["qc_min"] for r in ok)] if ok else None,
            "n_qc_below_0.65": sum(not r["qc_ok"] for r in ok), "n_touching_boundary": sum(r["touches_boundary"] for r in ok)}
    low = [s for s in CORTICAL if res[f"{s} / ICV"]["icc21"] < CORTICAL_FLOOR]
    out = {"meta": meta, "ratios": {k: v for k, v in res.items() if k.endswith("/ ICV")},
           "raw": {k: v for k, v in res.items() if not k.endswith("/ ICV")}, "cortical_below_floor": low}
    (HERE / "results" / "agreement.json").write_text(json.dumps(out, indent=1))
    L = ["# SynthSeg vs FreeSurfer 5.1: descriptive agreement (not a gate)", "",
         f"n = {meta['n_ok']} of 25 sampled train scans (failures: {meta['n_failed']}). SynthSeg QC min per scan "
         f"{meta['qc_min_range'][0]:.2f}-{meta['qc_min_range'][1]:.2f}; below 0.65: {meta['n_qc_below_0.65']}; "
         f"segmentations touching the patch boundary: {meta['n_touching_boundary']}.", "",
         "| Quantity | ICC(2,1) [95% CI] | Bias, % of FS mean | 95% limits of agreement, % | Pearson r |",
         "|---|---|---|---|---|"]
    for k, v in res.items():
        L.append(f"| {k} | {v['icc21']:.2f} [{v['icc21_ci95'][0]:.2f}, {v['icc21_ci95'][1]:.2f}] | "
                 f"{v['bias_pct_of_fs_mean']:+.1f} | {v['loa_low_pct']:+.0f} to {v['loa_high_pct']:+.0f} | "
                 f"{v['pearson_r']:.2f} |")
    L += ["", f"Cortical ratios with ICC(2,1) below {CORTICAL_FLOOR}: {', '.join(low) or 'none'}."]
    (HERE / "results" / "agreement.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
