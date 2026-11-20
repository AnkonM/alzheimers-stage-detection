"""Run SynthSeg on the sampled native train scans (see PLAN.md). Run alone, with LYP/.venv-app from the repo root:

    python -m analysis.volume_agreement.run

Resumable: a subject with private/<rid>/result.json is skipped. Failures are written, never replaced.
Reads RID, DX_bl, Image ID from selected_scans.csv, train RIDs, and the FreeSurfer volumes from step2 (no labels
are used beyond the DX_bl stratum quota fixed in PLAN.md).
"""
import csv
import glob
import json
import os
import pathlib

import numpy as np

from src.assessment.volumes import synthseg

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
PRIV = HERE / "private"
QUOTA = {"CN": 7, "EMCI": 6, "LMCI": 6, "AD": 6}
VOLS = ["Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV"]


def pool():
    sel = {r["Image ID"]: r for r in csv.DictReader(open(ROOT / "dataset/mri_raw/selected_scans.csv"))}
    train = {int(r["RID"]) for r in csv.DictReader(open(ROOT / "dataset/processed/train_rids.csv"))}
    fs = {int(float(r["RID"])): {k: float(r[k]) for k in VOLS}
          for r in csv.DictReader(open(ROOT / "dataset/processed/step2_features.csv"))
          if all(r[k] for k in VOLS)}
    out = []
    for p in sorted(glob.glob(os.environ.get("ADNI_NIFTI_GLOB", "adni_nifti/ADNI/*/*/*/I*/*.nii"))):
        i = p.split("/I")[-1].split("/")[0]
        if i in sel and int(sel[i]["RID"]) in train and int(sel[i]["RID"]) in fs:
            out.append({"rid": int(sel[i]["RID"]), "dx": sel[i]["DX_bl"], "path": p, "fs": fs[int(sel[i]["RID"])]})
    return sorted(out, key=lambda x: x["rid"])


def sample(p):
    rng = np.random.default_rng(0)
    chosen = []
    for dx, n in QUOTA.items():
        cand = [x for x in p if x["dx"] == dx]
        idx = sorted(rng.choice(len(cand), n, replace=False)) if len(cand) > n else range(len(cand))
        chosen += [cand[i] for i in idx]
    return chosen


def main():
    chosen = sample(pool())
    assert len(chosen) == 25, len(chosen)
    PRIV.mkdir(exist_ok=True)
    for i, c in enumerate(chosen, 1):
        d = PRIV / str(c["rid"])
        if (d / "result.json").exists():
            continue
        d.mkdir(exist_ok=True)
        try:
            r = synthseg.run(c["path"], str(d / "ss"))
            r = {k: v for k, v in r.items() if k != "seg_path"}
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {e}"}
        (d / "result.json").write_text(json.dumps({"rid": c["rid"], "fs": c["fs"], **r}))
        print(f"[{i}/25] done", flush=True)


if __name__ == "__main__":
    main()
