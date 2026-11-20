"""ALZ and the brain-age reliability rule per scan (run with LYP/.venv-mri). Frozen CNNs, inference only.

Mirrors analysis/fusion_v3/scan_inputs.py: per cross-fit fold model k (best.pt), the 3-seed mean
sigmoid(ALZ logit). Assignment (PLAN.md 3.3): TRAIN -> the out-of-fold model (fold from
alz_crossfit.csv); TEST / VAL / NOT_IN_COHORT -> mean of the 5 fold means.
--gate      G1b ALZ part + ALZ provenance: original baseline volumes (dataset/mri_prep/vol96) of the
            123 TEST subjects (vs scan_inputs_test.csv) and of the first 20 TRAIN and 20 VAL subjects
            of alz_crossfit.csv (vs alz_crossfit.csv ALZ).
--ingestion every usable scan of private/scans.csv -> private/scan_inputs.csv.
"""
import argparse
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "mri"))
sys.path.insert(0, os.path.join(ROOT, "analysis", "fusion_feasibility"))
from predict_alz import load_model, score  # noqa: E402
import m1_preprocess as m1  # noqa: E402

RUNS = os.path.join(ROOT, "analysis", "fusion_feasibility", "runs")
PRIVATE = os.path.join(HERE, "private")


def load(path):
    vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    assert vol.shape == (96, 96, 96) and vol.min() >= -1e-4 and vol.max() <= 1 + 1e-4, path
    return torch.from_numpy(vol).unsqueeze(0)


def fold_means(paths, device):
    """(n, 5): per fold model, 3-seed mean sigmoid(ALZ logit)."""
    vols = torch.stack([load(p) for p in paths])
    out = np.empty((len(paths), 5))
    for k in range(5):
        seeds = []
        for seed in (42, 43, 44):
            model, _ = load_model(os.path.join(RUNS, f"fold{k}_s{seed}", "best.pt"), device)
            seeds.append(score(model, vols, device)[1])
            del model
        out[:, k] = np.mean(seeds, 0)
    return out


def assign(fm, group, fold):
    alz = fm.mean(1)
    tr = (group == "TRAIN").to_numpy()
    alz[tr] = fm[np.flatnonzero(tr), fold[tr].astype(int)]
    return alz


def reliability(mni_paths):
    _, mask_w, _, _, _ = m1.load_template(96)
    mask = mask_w.numpy() > 0.5
    rows = []
    for p in mni_paths:
        x = nib.load(p).get_fdata(dtype="float32")[mask]
        rows.append({"n_zero": int((x == 0).sum()), "sat_frac": float((x >= 0.999).mean())})
    r = pd.DataFrame(rows)
    r["unreliable"] = (r.n_zero == 0) | (r.sat_frac > 0.10)
    return r


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gate", action="store_true")
    g.add_argument("--ingestion", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cf = pd.read_csv(os.path.join(ROOT, "analysis", "fusion_feasibility", "alz_crossfit.csv"))
    fold = cf.set_index("RID").fold
    if a.gate:
        ref = pd.read_csv(os.path.join(ROOT, "analysis", "fusion_v3", "scan_inputs_test.csv")).dropna(subset=["ALZ"])
        prov = pd.concat([cf[cf.clin_split == s].head(20) for s in ("train", "val")])
        d = pd.concat([ref[["RID", "PTID"]].assign(group="TEST"),
                       prov[["RID", "PTID"]].assign(group=prov.clin_split.str.upper())], ignore_index=True)
        prep = os.path.join(ROOT, "dataset", "mri_prep")
        fm = fold_means([os.path.join(prep, "vol96", f"{p}.nii.gz") for p in d.PTID], device)
        d["ALZ"] = assign(fm, d.group, d.RID.map(fold))
        t = d[d.group == "TEST"].merge(ref[["RID", "ALZ", "unreliable"]].rename(columns={"ALZ": "ALZ_ref", "unreliable": "unreliable_ref"}), on="RID")
        dt = float((t.ALZ - t.ALZ_ref).abs().max())
        rel = reliability([os.path.join(prep, "mni", f"{p}.nii.gz") for p in t.PTID])
        same_rel = bool((rel.unreliable.to_numpy() == t.unreliable_ref.astype(bool).to_numpy()).all())
        print(f"G1b ALZ: TEST n={len(t)} max |ALZ - scan_inputs_test| = {dt:.2e} (pass {dt <= 1e-5}); "
              f"reliability flags identical {same_rel}")
        for s in ("TRAIN", "VAL"):
            p = d[d.group == s].merge(cf[["RID", "ALZ"]], on="RID", suffixes=("", "_ref"))
            dp = float((p.ALZ - p.ALZ_ref).abs().max())
            print(f"ALZ provenance {s}: n={len(p)} max |ALZ - alz_crossfit| = {dp:.2e} (pass {dp <= 1e-5})")
        return
    s = pd.read_csv(os.path.join(PRIVATE, "scans.csv"), low_memory=False)
    s = s[s.usable].reset_index(drop=True)
    prep = os.path.join(ROOT, "dataset", "mri_prep_patha")
    fm = np.concatenate([fold_means([os.path.join(prep, "vol96", f"{k}.nii.gz") for k in s.prep_key[i:i + 256]],
                                    device) for i in range(0, len(s), 256)])
    out = pd.DataFrame({"key": s.prep_key, "RID": s.RID, "group": s.group,
                        **{f"alz_fold{k}": fm[:, k] for k in range(5)}})
    out["alz_rule"] = np.where(out.group == "TRAIN", "out-of-fold model", "mean of 5 fold models")
    out["ALZ"] = assign(fm, out.group, out.RID.map(fold))
    assert out.ALZ.notna().all(), "TRAIN subject without a cross-fit fold"
    rel = reliability([os.path.join(prep, "mni", f"{k}.nii.gz") for k in s.prep_key])
    out = pd.concat([out, rel], axis=1)
    out.to_csv(os.path.join(PRIVATE, "scan_inputs.csv"), index=False)
    print(f"ALZ for {len(out)} scans; unreliable brain-age scans: {int(out.unreliable.sum())}")


if __name__ == "__main__":
    main()
