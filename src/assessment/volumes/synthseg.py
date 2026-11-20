"""SynthSeg volumes for new (non-ADNI-table) scans. Runs FreeSurfer 7.4.1 mri_synthseg 2.0 in .venv-synthseg.

Run with nothing else memory-heavy resident: measured peak up to 13.61 GiB RSS (13,231,404 kB, 16 threads) on a 20 GB VM;
the batch runner stops the mri and dbn workers first and checks a 15.2 GiB MemAvailable floor. Same invocation as analysis/mri_morphometry/run_synthseg.py
(--parc --cpu --crop 160 192 160). Label mapping as analysis/mri_morphometry/smoke_analysis.py.
Input is the native T1 (what an upload is), not the m1 output.
"""
import csv
import os
import subprocess

import nibabel as nib
import numpy as np

from ..paths import LYP

FSHOME = LYP / "third_party" / "fs_synthseg"
PYTHON = LYP / ".venv-synthseg" / "bin" / "python"
PATCH = (160, 192, 160)
QC_MIN = 0.65  # SynthSeg 2.0 paper flagging cut-off (also quoted in the earlier smoke report)
VENT = ["left lateral ventricle", "right lateral ventricle", "left inferior lateral ventricle",
        "right inferior lateral ventricle"]
NON_BRAIN = VENT + ["3rd ventricle", "4th ventricle", "csf", "total intracranial"]


class SynthSegError(RuntimeError):
    pass


def volumes_from_row(v: dict) -> dict:
    """Map a SynthSeg --vol row (label name -> mm^3) to the six model inputs plus ICV."""
    f = {k: float(x) for k, x in v.items() if k != "subject"}
    out = {"Hippocampus": f["left hippocampus"] + f["right hippocampus"],
           "Ventricles": sum(f[k] for k in VENT), "ICV": f["total intracranial"]}
    for name, lab in (("Entorhinal", "entorhinal"), ("Fusiform", "fusiform"), ("MidTemp", "middletemporal")):
        out[name] = f[f"ctx-lh-{lab}"] + f[f"ctx-rh-{lab}"]
    out["WholeBrain"] = sum(x for k, x in f.items() if not k.startswith("ctx-") and k not in NON_BRAIN)
    return out


def touches_patch_boundary(seg_path: str) -> bool:
    """True if brain labels reach the edge of the segmentation volume (possible clipping by --crop)."""
    seg = np.asarray(nib.load(seg_path).dataobj)
    brain = seg > 0
    return any(brain.take(i, axis=a).any() for a in range(3) for i in (0, brain.shape[a] - 1))


def run(native_path: str, work_dir: str, threads: int = 16) -> dict:
    """Segment one native scan. Returns volumes (mm^3), min QC score and the clipping flag."""
    os.makedirs(work_dir, exist_ok=True)
    seg, vol, qc = (os.path.join(work_dir, n) for n in ("seg.nii.gz", "vol.csv", "qc.csv"))
    cmd = [str(PYTHON), str(FSHOME / "bin" / "mri_synthseg"), "--i", native_path, "--o", seg, "--parc", "--cpu",
           "--threads", str(threads), "--crop", *map(str, PATCH), "--vol", vol, "--qc", qc]
    r = subprocess.run(cmd, env={**os.environ, "FREESURFER_HOME": str(FSHOME)}, capture_output=True, text=True)
    if r.returncode or not os.path.exists(vol):
        raise SynthSegError(f"mri_synthseg failed (exit {r.returncode}): {r.stderr[-300:]}")
    row = next(csv.DictReader(open(vol)))
    q = next(csv.DictReader(open(qc)))
    qcs = [float(x) for k, x in q.items() if k != "subject"]
    return {"volumes": volumes_from_row(row), "qc_min": min(qcs), "qc_ok": min(qcs) >= QC_MIN,
            "touches_boundary": touches_patch_boundary(seg), "seg_path": seg}
