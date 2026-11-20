"""Why does the engine's LayerCAM region density differ slightly from the old card's (demo/card_data/cam.npz)?

Run with LYP/.venv-mri from the repo root:  python tests/diagnostics/cam_density_batch_diag.py <PTID>   (local data)

Recomputes the subject's densities three ways with the frozen CNNs and src/card/build_cam.py's own arithmetic:
  card batching    each model scores the subject inside the card's chunk of 32 subjects (cams batch size 8),
  single scan      the subject alone (how the engine runs),
  float64 vs float32 accumulation of the region masses.
and prints the max |difference| to the stored card values. Reads only the subject's volume; no labels, no metrics.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "analysis", "explainability"))
import common as C  # noqa: E402
import p1_lib as L  # noqa: E402

import json  # noqa: E402

ptid = sys.argv[1]
stage = json.load(open(os.path.join(C.RESULTS, "card_rules.json")))["cnn_stage"]["stage"]
atl, info = L.atlas()
regions, vox, nb = info["regions"], info["voxels"], info["brain_voxels"]
mask = L.brain_mask()
d = L.subjects(include_test=True)
me = int(np.flatnonzero(d.PTID.to_numpy() == ptid)[0])
reg_idx = [np.flatnonzero(atl.ravel() == i + 1) for i in range(len(regions))]
card = np.load("demo/card_data/cam.npz", allow_pickle=True)
stored = card["density"][int(np.flatnonzero(card["ptid"] == ptid)[0])]


def density(mass):
    share = mass / mass.sum()
    return share / (np.array([vox[r] for r in regions]) / nb)


mass_card32 = np.zeros(len(regions))
mass_card64 = np.zeros(len(regions))
mass_single = np.zeros(len(regions))
n = 0
for mid in range(len(L.MODELS)):
    model = L.get_model(mid)
    rows = [i for i, r in enumerate(d.itertuples()) if mid in L.models_of(r)]
    for c in range(0, len(rows), 32):
        chunk = rows[c:c + 32]
        if me in chunk:
            vols = L.load_volumes(list(d.path.iloc[chunk]))
            maps, _ = L.cams(model, vols, stages=[stage])
            f = L.upsample(maps[("layercam", stage)], mask)[chunk.index(me)].ravel()
            mass_card32 += [f[ix].sum() for ix in reg_idx]                        # card arithmetic (float32 sums)
            mass_card64 += [f[ix].astype(np.float64).sum() for ix in reg_idx]
    vol = L.load_volumes([d.path.iloc[me]])
    maps, _ = L.cams(model, vol, stages=[stage])
    f = L.upsample(maps[("layercam", stage)], mask)[0].ravel()
    mass_single += [f[ix].astype(np.float64).sum() for ix in reg_idx]
    n += 1
    del model
    torch.cuda.empty_cache()
print(f"models {n}; stored card density vs:")
print(f"  card batching, card arithmetic    max|diff| {np.abs(density(mass_card32) - stored).max():.2e}")
print(f"  card batching, float64 sums       max|diff| {np.abs(density(mass_card64) - stored).max():.2e}")
print(f"  single scan, float64 sums         max|diff| {np.abs(density(mass_single) - stored).max():.2e}")
print(f"  single scan vs card batching      max|diff| {np.abs(density(mass_single) - density(mass_card64)).max():.2e}")
