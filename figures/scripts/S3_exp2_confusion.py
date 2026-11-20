"""Supplementary S3: Experiment 2 seed-averaged CV confusion matrices, C (clinical only) vs F-KAN, for the
Full-spec, T1 and T2 tiers (train CV, n = 569).

Inputs (read-only from the committed branch, via git show):
  exp/reduced-clinical:experiments/exp2_reduced_clinical/tables/confusion_cv_{Full-spec,T1,T2}_{C,F-KAN}.csv
Every drawn cell annotation is read back from the figure and asserted equal to the CSV.
Run with LYP/.venv-mri.
"""
import io
import json

import numpy as np
import pandas as pd

from _prov import OUT, Prov
import paper_style as S

P = Prov("S3_exp2_confusion", __file__)
REF = "exp/reduced-clinical"
TIERS = ["Full-spec", "T1", "T2"]
SYSTEMS = [("C", "C (clinical only)"), ("F-KAN", "F-KAN (clinical + MRI)")]
NAMES = ["CN", "EMCI", "LMCI", "AD"]
M = {}
for t in TIERS:
    for s, _ in SYSTEMS:
        df = pd.read_csv(io.BytesIO(P.git_blob(REF, f"experiments/exp2_reduced_clinical/tables/confusion_cv_{t}_{s}.csv")), index_col=0)
        P.check(f"{t} {s}: labels", list(df.index) == [f"true_{n}" for n in NAMES] and list(df.columns) == [f"pred_{n}" for n in NAMES],
                None, [list(df.index), list(df.columns)])
        M[(t, s)] = df.to_numpy()
        P.check(f"{t} {s}: n == 569", int(M[(t, s)].sum()) == 569, 569, int(M[(t, s)].sum()))

import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(2, 3, figsize=(S.DOUBLE, 5.0))
fig.subplots_adjust(left=0.10, right=0.99, top=0.95, bottom=0.08, wspace=0.12, hspace=0.30)
ok = 0
for c, t in enumerate(TIERS):
    for r, (s, lab) in enumerate(SYSTEMS):
        ax, m = axes[r, c], M[(t, s)]
        ax.imshow(m / m.sum(1, keepdims=True), cmap=S.SEQ_CMAP, vmin=0, vmax=1)
        txt = {}
        for i in range(4):
            for j in range(4):
                txt[(i, j)] = ax.text(j, i, str(int(m[i, j])), ha="center", va="center", fontsize=8,
                                      color="white" if m[i, j] / m[i].sum() > 0.55 else "black")
        ok += sum(int(txt[(i, j)].get_text()) == int(m[i, j]) for i in range(4) for j in range(4))
        ax.set_xticks(range(4), NAMES if r == 1 else [""] * 4)
        ax.set_yticks(range(4), NAMES if c == 0 else [""] * 4)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if r == 0:
            ax.text(0.5, 1.03, t, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
        if c == 0:
            ax.set_ylabel(f"{lab}\nTrue stage")
        if r == 1:
            ax.set_xlabel("Predicted stage")
P.check("every drawn cell == CSV (6 matrices x 16 cells)", ok == 96, 96, ok)
P.d["numbers"] = {f"{t}/{s}": M[(t, s)].tolist() for t in TIERS for s, _ in SYSTEMS}
outs = S.save(fig, OUT, "S3_exp2_confusion")
P.write(outs)
print("cells verified:", ok)
