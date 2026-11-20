"""Figure 1: (a) the CNN's actual 96^3 input for one ADNI training participant, (b) the 12-region atlas on the
MNI template at the same slices. Run with LYP/.venv-mri.

Selection rule (fixed in code, applied to tables before any image is opened):
  rows of dataset/mri_prep/mri_index.csv with split == "train", DX_bl == "CN" and RID in
  dataset/processed/train_rids.csv; m1 preprocessing ok == True and registration verdict PASS
  (corr_vs_mni >= 0.30, src/assessment/mri/qc.py CORR_PASS; the m1_report_fix12.csv row overrides
  m1_report_full.csv when present); sort by PTID ascending; take the first.
Only the internal row index is recorded, never the subject ID or file name.
"""
import io
import json
import os
import re

import nibabel as nib
import numpy as np
import pandas as pd

from _prov import MAIN, OUT, WT, Prov
import paper_style as S

P = Prov("fig1_mri_input", __file__)
CORR_PASS = 0.30
src = open(os.path.join(WT, "src/assessment/mri/qc.py")).read()
m = re.search(r"CORR_PASS, CORR_FAIL = ([0-9.]+), ([0-9.]+)", src)
P.check("registration PASS threshold equals src/assessment/mri/qc.py CORR_PASS", m and float(m.group(1)) == CORR_PASS,
        CORR_PASS, m and float(m.group(1)))

# ---- 1. selection, tables only -------------------------------------------------------------
idx = pd.read_csv(io.BytesIO(P.file("dataset/mri_prep/mri_index.csv")))
rep = pd.read_csv(io.BytesIO(P.file("dataset/mri_prep/m1_report_full.csv")))
fix = pd.read_csv(io.BytesIO(P.file("dataset/mri_prep/m1_report_fix12.csv")))
train = set(pd.read_csv(io.BytesIO(P.file("dataset/processed/train_rids.csv"))).RID.astype(int))
qc = pd.concat([rep[~rep.PTID.isin(fix.PTID)], fix])[["PTID", "ok", "corr_vs_mni", "frac_outside_mask"]]
cand = idx.reset_index().rename(columns={"index": "row"})
cand = cand[(cand.split == "train") & (cand.DX_bl == "CN") & cand.RID.astype(int).isin(train)].merge(qc, on="PTID")
cand = cand[(cand.ok.astype(str) == "True") & (cand.corr_vs_mni >= CORR_PASS)].sort_values("PTID")
P.check("at least one eligible subject", len(cand) > 0, ">0", len(cand))
sel = cand.iloc[0]
P.d["selection"] = {
    "rule": "first CN subject of the training split (mri_index split == train, RID in train_rids.csv) in ascending "
            "PTID order with m1 ok == True and registration verdict PASS (corr_vs_mni >= 0.30; fix12 report row "
            "overrides the full report when present)",
    "eligible_n": int(len(cand)), "mri_index_row": int(sel.row), "corr_vs_mni": round(float(sel.corr_vs_mni), 4),
    "registration_verdict": "PASS", "report_source": "m1_report_fix12.csv" if sel.PTID in set(fix.PTID) else "m1_report_full.csv"}

# ---- 2. load the exact CNN input (main checkout, gitignored, read-only) ----------------------
vpath = os.path.join(MAIN, *str(sel.vol_path).replace("\\", "/").split("/"))
vb = P.file(vpath, label="dataset/mri_prep/vol96/<redacted>.nii.gz (main checkout, gitignored; == mri_index vol_path)")
img = nib.Nifti1Image.from_bytes(__import__("gzip").decompress(vb))
vol = np.asarray(img.dataobj, dtype=np.float32)
tb = P.file(os.path.join(MAIN, "analysis/explainability/atlas/work/template96.nii.gz"),
            label="analysis/explainability/atlas/work/template96.nii.gz (main checkout, gitignored)")
timg = nib.Nifti1Image.from_bytes(__import__("gzip").decompress(tb))
tmpl = np.asarray(timg.dataobj, dtype=np.float32)
aimg = nib.Nifti1Image.from_bytes(__import__("gzip").decompress(P.file("analysis/explainability/atlas/atlas96.nii.gz")))
atlas = np.asarray(aimg.dataobj).astype(int)
regions = json.loads(P.file("analysis/explainability/atlas/regions.json"))
sl = regions["card_slices"]

P.check("CNN input shape 96^3", vol.shape == (96, 96, 96), [96, 96, 96], list(vol.shape))
P.check("CNN input intensity in [0, 1]", vol.min() >= -1e-6 and vol.max() <= 1 + 1e-6, "[0,1]",
        [float(vol.min()), float(vol.max())])
codes = {"vol96": "".join(nib.aff2axcodes(img.affine)), "template96": "".join(nib.aff2axcodes(timg.affine)),
         "atlas96": "".join(nib.aff2axcodes(aimg.affine))}
P.check("voxel axes point to (S, P, R) for vol96, template96 and atlas96 (nib.aff2axcodes)",
        set(codes.values()) == {"SPR"}, "SPR", codes)
P.check("vol96, template96 and atlas96 share one grid (affine atol 1e-3)",
        np.allclose(img.affine, timg.affine, atol=1e-3) and np.allclose(img.affine, aimg.affine, atol=1e-3), True,
        bool(np.allclose(img.affine, timg.affine, atol=1e-3)))
m1 = open(os.path.join(WT, "src/mri/m1_preprocess.py")).read()
P.check("m1 ORIENTATION constant is 'IAL' (ITK 'from' codes: index increases towards S, P, R)",
        'ORIENTATION = "IAL"' in m1, "IAL", re.search(r'ORIENTATION = "(\w+)"', m1).group(1))

# ---- 3. brain-only check (decision 2026-11-18): no nonzero voxel outside the template support ---
support = tmpl > 0
mask05 = atlas > 0  # == template mask resampled to 96^3 and thresholded at 0.5 (atlas_build.py)
nz = vol > 1e-6
out_support = int((nz & ~support).sum())
P.d["brain_only"] = {"nonzero_voxels": int(nz.sum()), "nonzero_outside_template_support": out_support,
                     "nonzero_outside_0.5_mask_rim": int((nz & ~mask05).sum()),
                     "rim_fraction": round(float((nz & ~mask05).sum() / nz.sum()), 4),
                     "visual_check": "pending (done by eye on the rendered PNG; recorded in README)"}
P.check("brain-only: 0 nonzero voxels outside the template brain support", out_support == 0, 0, out_support)

# ---- 4. plot -----------------------------------------------------------------------------------
sp = np.sqrt((img.affine[:3, :3] ** 2).sum(0))  # mm per voxel along (S, P, R)
bb = np.argwhere(support)
lo, hi = np.maximum(bb.min(0) - 1, 0), np.minimum(bb.max(0) + 2, 96)


def views(a):
    """(2-D array, origin, aspect, show L/R) for axial, coronal and left sagittal; neurological convention."""
    ax_ = a[sl["axial_S"], lo[1]:hi[1], lo[2]:hi[2]]          # rows P (anterior on top), cols R (subject left on the left)
    co_ = a[lo[0]:hi[0], sl["coronal_P"], lo[2]:hi[2]]        # rows S (origin lower), cols R
    sa_ = a[lo[0]:hi[0], lo[1]:hi[1], sl["sagittal_R"]]       # rows S, cols P (anterior on the left)
    return [(ax_, "upper", sp[1] / sp[2], True), (co_, "lower", sp[0] / sp[2], True), (sa_, "lower", sp[0] / sp[1], False)]


import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

widths = [v[0].shape[1] / v[2] / v[0].shape[0] for v in views(vol)]  # display width per unit height
fig = plt.figure(figsize=(S.DOUBLE, 5.05))
gs = fig.add_gridspec(3, 3, width_ratios=widths, height_ratios=[1, 1, 0.30], hspace=0.06, wspace=0.03,
                      left=0.02, right=0.98, top=0.99, bottom=0.0)
names = regions["regions"]
cmap = ListedColormap(["#000000"] + S.ATLAS_12 + [S.OTHER_GREY])
for c, (img2, origin, aspect, lr) in enumerate(views(vol)):
    ax = fig.add_subplot(gs[0, c])
    ax.imshow(img2, cmap="gray", vmin=0, vmax=1, origin=origin, aspect=aspect, interpolation="nearest")
    ax.set_axis_off()
    if lr:
        ax.text(0.03, 0.97, "L", transform=ax.transAxes, color="white", fontsize=8, va="top", ha="left")
        ax.text(0.97, 0.97, "R", transform=ax.transAxes, color="white", fontsize=8, va="top", ha="right")
    if c == 0:
        S.panel_label(ax, "a", x=0.0, y=1.0)
for c, ((t2, origin, aspect, lr), (a2, _, _, _)) in enumerate(zip(views(tmpl), views(atlas))):
    ax = fig.add_subplot(gs[1, c])
    ax.imshow(t2, cmap="gray", vmin=0, vmax=float(tmpl.max()), origin=origin, aspect=aspect, interpolation="nearest")
    ax.imshow(np.ma.masked_where(a2 == 0, a2), cmap=cmap, vmin=0, vmax=len(names), alpha=0.6, origin=origin,
              aspect=aspect, interpolation="nearest")
    ax.set_axis_off()
    if lr:
        ax.text(0.03, 0.97, "L", transform=ax.transAxes, color="white", fontsize=8, va="top", ha="left")
        ax.text(0.97, 0.97, "R", transform=ax.transAxes, color="white", fontsize=8, va="top", ha="right")
    if c == 0:
        S.panel_label(ax, "b", x=0.0, y=1.0)
lax = fig.add_subplot(gs[2, :])
lax.set_axis_off()
lax.legend([Patch(color=cmap(i + 1)) for i in range(len(names))], names, loc="center", ncol=5, fontsize=8,
           handlelength=1.0, columnspacing=1.0, handletextpad=0.4)
P.check("atlas legend has 12 named regions + other", len(names) == 13 and names[-1] == "other", 13, len(names))
P.d["display"] = {"slices": sl, "convention": "neurological (subject left on image left; anterior up in axial)",
                  "voxel_mm_SPR": [round(float(x), 3) for x in sp], "crop_to_template_support_bbox": [lo.tolist(), hi.tolist()],
                  "intensity_window": "0-1 (array range, unaltered)"}
outs = S.save(fig, OUT, "fig1_mri_input")
P.write(outs)
print(json.dumps({"selection_row": int(sel.row), "eligible": int(len(cand)), "brain_only": P.d["brain_only"], "axcodes": codes}))
