"""Domain-shift diagnostics: OASIS-1 vs the ADNI training scans, summary statistics only.

Read-only. Per-scan features are computed in memory from the m1 outputs (`vol96`, `mni`) and only aggregates
leave this module: distribution summaries, a standardised mean difference and a two-sample KS statistic against
ADNI, QC verdict counts, and acquisition-parameter counts. Nothing here feeds a model or changes a result.

Definitions (identical code for both datasets, so the comparison is like for like):
  vol96 (96^3, min-max scaled, brain-masked by the fixed MNI mask in m1):
    nonzero_frac   fraction of voxels above 1e-6
    nz_mean/nz_std mean and SD of the nonzero voxels
    p05..p99       percentiles of the nonzero voxels
  mni (189 x 233 x 164 template space):
    mask_volume_mm3  number of voxels above 1e-6 times the voxel volume from the image header. NOTE: after the
                     affine registration and the fixed template mask this is tissue-filled template-space volume,
                     not native brain volume (there is no ICV information in it).
"""
import numpy as np
from scipy import stats as sps

from ..assessment.mri.qc import CORR_FAIL, CORR_PASS

EPS = 1e-6
PCTS = (5, 25, 50, 75, 95, 99)


def vol96_features(a):
    a = np.asarray(a, float)
    nz = a[a > EPS]
    out = {"nonzero_frac": float((a > EPS).mean()), "frac_zero": float((a == 0).mean())}
    if nz.size:
        out.update({"nz_mean": float(nz.mean()), "nz_std": float(nz.std())})
        out.update({f"p{p:02d}": float(np.percentile(nz, p)) for p in PCTS})
    else:
        out.update({"nz_mean": float("nan"), "nz_std": float("nan"), **{f"p{p:02d}": float("nan") for p in PCTS}})
    return out


def mask_volume_mm3(mni, zooms):
    return float((np.asarray(mni) > EPS).sum() * float(np.prod(zooms[:3])))


def case_features(vol96_path, mni_path, corr=None):
    import nibabel as nib
    v = nib.load(str(vol96_path))
    m = nib.load(str(mni_path))
    f = vol96_features(np.asarray(v.dataobj))
    f["mask_volume_mm3"] = mask_volume_mm3(np.asarray(m.dataobj), m.header.get_zooms())
    if corr is not None:
        f["corr_vs_mni"] = float(corr)
    return f


def verdict(corr):
    """Registration verdict by the engine rule (qc.py): PASS >= 0.30, WARN 0.15-0.30, FAIL below 0.15."""
    return "PASS" if corr >= CORR_PASS else "WARN" if corr >= CORR_FAIL else "FAIL"


def qc_counts(corrs, groups=None):
    out = {"PASS": 0, "WARN": 0, "FAIL": 0}
    by = {}
    for i, c in enumerate(corrs):
        v = verdict(c)
        out[v] += 1
        if groups is not None:
            by.setdefault(str(groups[i]), {"PASS": 0, "WARN": 0, "FAIL": 0})[v] += 1
    res = {"all": out, "n": int(len(corrs))}
    if groups is not None:
        res["by_group"] = by
    return res


def summarize(values):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "p05": float(np.percentile(v, 5)), "p25": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "p75": float(np.percentile(v, 75)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def compare(new, ref):
    """Shift of `new` relative to `ref`: standardised mean difference (pooled SD), median difference and the
    two-sample KS statistic with its p-value. Summary numbers only."""
    a, b = (np.asarray(x, float) for x in (new, ref))
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    ks = sps.ks_2samp(a, b)
    return {"smd": float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0,
            "median_diff": float(np.median(a) - np.median(b)), "ks_stat": float(ks.statistic),
            "ks_p": float(ks.pvalue), "n_new": int(len(a)), "n_ref": int(len(b))}


def collect(features):
    """list of per-case dicts -> dict of column arrays (kept in memory only)."""
    keys = sorted({k for f in features for k in f})
    return {k: np.array([f.get(k, np.nan) for f in features], float) for k in keys}


def acquisition_counts(rows):
    """rows: dicts of scan parameters (e.g. TR, TE, TI, flip, voxel size, shape) -> value counts per parameter."""
    out = {}
    for r in rows:
        for k, v in r.items():
            out.setdefault(k, {})
            out[k][str(v)] = out[k].get(str(v), 0) + 1
    return out


def adni_reference(base, index_csv, report_csv, split="train"):
    """ADNI train-split per-scan features from the stored m1 outputs, computed with the same code as OASIS.
    `base` is dataset/mri_prep. Returns column arrays in memory (never written)."""
    import pandas as pd
    idx = pd.read_csv(index_csv)
    idx = idx[idx["split"] == split]
    rep = pd.read_csv(report_csv).set_index("PTID")
    feats = []
    for ptid in idx["PTID"]:
        if ptid not in rep.index or str(rep.loc[ptid, "ok"]).strip().lower() != "true":
            continue
        feats.append(case_features(f"{base}/vol96/{ptid}.nii.gz", f"{base}/mni/{ptid}.nii.gz",
                                   corr=rep.loc[ptid, "corr_vs_mni"]))
    return collect(feats)


def domain_shift_report(new_cols, ref_cols, acquisition=None, groups=None):
    """Aggregate JSON-able dict. `new_cols` / `ref_cols` come from collect(); `groups` optionally labels each new
    case (class name) for QC counts by group."""
    res = {"features": {}, "qc_new": qc_counts(new_cols["corr_vs_mni"], groups) if "corr_vs_mni" in new_cols else None,
           "qc_reference": qc_counts(ref_cols["corr_vs_mni"]) if "corr_vs_mni" in ref_cols else None,
           "definitions": __doc__.split("Definitions")[1].strip()}
    for k in sorted(set(new_cols) & set(ref_cols)):
        res["features"][k] = {"new": summarize(new_cols[k]), "reference": summarize(ref_cols[k]),
                              "shift": compare(new_cols[k], ref_cols[k])}
    if acquisition is not None:
        res["acquisition_new"] = acquisition_counts(acquisition)
    return res
