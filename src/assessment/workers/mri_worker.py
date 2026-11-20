"""MRI stage worker (LYP/.venv-mri): ALZ from the 15 frozen CNNs and the frozen intensity-reliability rule.

Reuses analysis/temporal_eval/scan_alz.py (volume loader, reliability rule) and
analysis/fusion_feasibility/predict_alz.py (checkpoint loader, scorer) unchanged.
ALZ for any scan outside the training set = mean over the 5 fold models of the 3-seed mean sigmoid(ALZ logit),
the TEST/VAL rule of scan_inputs.py. Models are loaded once and kept on the device.
"""
import os
import sys

import json
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "analysis", "temporal_eval"))
import scan_alz as SA  # noqa: E402
from predict_alz import load_model, score  # noqa: E402

from src.assessment.workers.client import serve  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODELS = {}


def models():
    if not _MODELS:
        for k in range(5):
            for seed in (42, 43, 44):
                _MODELS[k, seed] = load_model(os.path.join(SA.RUNS, f"fold{k}_s{seed}", "best.pt"), DEVICE)[0]
    return _MODELS


def alz(vol96_paths):
    vols = torch.stack([SA.load(p) for p in vol96_paths])
    m = models()
    fm = np.empty((len(vol96_paths), 5))
    for k in range(5):
        fm[:, k] = np.mean([score(m[k, s], vols, DEVICE)[1] for s in (42, 43, 44)], 0)
    return {"fold_means": fm.tolist(), "alz": fm.mean(1).tolist(), "device": str(DEVICE)}


def reliability(mni_paths):
    r = SA.reliability(mni_paths)
    return {"n_zero": r.n_zero.tolist(), "sat_frac": r.sat_frac.tolist(), "unreliable": r.unreliable.tolist()}


_EXPL = []


def _expl():
    if not _EXPL:
        sys.path.insert(0, os.path.join(ROOT, "analysis", "explainability"))
        import common as C
        import p1_lib as L
        _EXPL.extend([C, L])
    return _EXPL


def layercam(vol96_path, out_dir):
    """LayerCAM of the ALZ logit, analysis/explainability P1 method (src/card/build_cam.py): stage from card_rules.json,
    all 15 fold models for a case outside the training set, maps upsampled to 96^3 on the template brain with unit
    positive mass, averaged. Returns the 12 named-region attribution densities (share / voxel share) and writes
    overlay.nii.gz (map scaled to its maximum) on the vol96 grid."""
    import nibabel as nib
    C, L = _expl()
    stage = json.load(open(os.path.join(C.RESULTS, "card_rules.json")))["cnn_stage"]["stage"]
    atl, info = L.atlas()
    regions, vox, nb = info["regions"], info["voxels"], info["brain_voxels"]
    mask = L.brain_mask()
    vol = L.load_volumes([vol96_path])
    acc = np.zeros((96, 96, 96), np.float64)
    for mid in range(len(L.MODELS)):
        model = L.get_model(mid)
        maps, _ = L.cams(model, vol, stages=[stage])
        acc += L.upsample(maps[("layercam", stage)], mask)[0]
        del model
        torch.cuda.empty_cache()
    acc /= len(L.MODELS)
    flat = atl.ravel()
    mass = np.array([acc.ravel()[flat == i + 1].sum() for i in range(len(regions))])
    dens = (mass / mass.sum()) / (np.array([vox[r] for r in regions]) / nb)
    img = nib.load(vol96_path)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "overlay.nii.gz")
    nib.save(nib.Nifti1Image((acc / acc.max()).astype(np.float32), img.affine), path)
    return {"regions": list(regions), "density": dens.tolist(), "stage": stage, "n_models": len(L.MODELS),
            "overlay": path}


_M1 = []


def preprocess(native_path, out_dir):
    """m1 preprocessing, unchanged (src/mri/m1_preprocess): writes mni.nii.gz and vol96.nii.gz, returns QC metrics and
    the native header facts the voxel-size gate needs."""
    import ants
    import nibabel as nib
    sys.path.insert(0, os.path.join(ROOT, "src", "mri"))
    import m1_preprocess as m1
    if not _M1:
        _M1.append(m1.load_template(96))
    head_w, mask_w, brain_w, tmpl_o, mask_o = _M1[0]
    full, small = m1.preprocess(native_path, head_w, mask_w, brain_w, 96)
    os.makedirs(out_dir, exist_ok=True)
    for img, name in ((full, "mni.nii.gz"), (small, "vol96.nii.gz")):
        ants.image_write(img, os.path.join(out_dir, name))
    h = nib.load(native_path)
    return {"qc": m1.qc_metrics(small, tmpl_o, mask_o), "spacing": [float(x) for x in h.header.get_zooms()[:3]],
            "shape": [int(x) for x in h.shape], "mni": os.path.join(out_dir, "mni.nii.gz"),
            "vol96": os.path.join(out_dir, "vol96.nii.gz")}


if __name__ == "__main__":
    serve({"alz": alz, "reliability": reliability, "layercam": layercam, "preprocess": preprocess,
           "ping": lambda: "mri"})
