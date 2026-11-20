"""Supplementary S2: CNN (LayerCAM layer1) deletion and insertion faithfulness curves (re-plot of F0009).

Input: analysis/explainability/results/p5_cnn_curves.csv (committed run; not the dry-run cache).
Sets: train_AD and val_AD; curves: attribution-ordered vs random voxel order.
Run with LYP/.venv-mri.
"""
import io
import json

import numpy as np
import pandas as pd

from _prov import OUT, Prov
import paper_style as S

P = Prov("S2_faithfulness", __file__)
PATH = "analysis/explainability/results/p5_cnn_curves.csv"
P.check("input is the committed results file, not the dry-run cache", "cache/dry" not in PATH, True, True)
d = pd.read_csv(io.BytesIO(P.file(PATH)))
P.check("sets", sorted(d.set.unique()) == ["train_AD", "val_AD"], ["train_AD", "val_AD"], sorted(d.set.unique()))
P.check("curves", sorted(d.curve.unique()) == ["del_attr", "del_rand", "ins_attr", "ins_rand"], None, sorted(d.curve.unique()))

import matplotlib.pyplot as plt  # noqa: E402

COL = {"val_AD": S.OKABE_ITO["blue"], "train_AD": S.OKABE_ITO["vermillion"]}
LAB = {"val_AD": "Val AD", "train_AD": "Train AD"}
fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.9), sharey=False)
fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.30, wspace=0.25)
drawn = []
for ax, kind, xl in ((axes[0], "del", "Fraction of brain voxels replaced"),
                     (axes[1], "ins", "Fraction of brain voxels inserted")):
    for st in ("val_AD", "train_AD"):
        for order, ls in (("attr", "-"), ("rand", "--")):
            c = d[(d.set == st) & (d.curve == f"{kind}_{order}")].sort_values("fraction")
            ln, = ax.plot(c.fraction, c.mean_alz_prob, ls=ls, color=COL[st], marker="o", ms=2.2,
                          label=f"{LAB[st]}, {'attribution' if order == 'attr' else 'random'} order")
            drawn.append(np.array_equal(ln.get_ydata(), c.mean_alz_prob.to_numpy()))
    ax.set_xlabel(xl)
    ax.set_ylabel("Mean CNN AD probability")
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, handlelength=2.0, columnspacing=1.5, bbox_to_anchor=(0.5, 0.0))
S.panel_label(axes[0], "a", x=-0.12, y=1.04)
S.panel_label(axes[1], "b", x=-0.12, y=1.04)
P.check("every plotted curve == CSV", all(drawn) and len(drawn) == 8, 8, sum(drawn))
end = d[d.fraction == d.fraction.max()].set_index(["set", "curve"]).mean_alz_prob.round(4)
P.d["numbers"] = {"start": d[d.fraction == 0].set_index(["set", "curve"]).mean_alz_prob.round(4).astype(float).to_dict(),
                  "end": end.astype(float).to_dict()}
P.d["numbers"] = {k: {f"{a}/{b}": v for (a, b), v in x.items()} for k, x in P.d["numbers"].items()}
outs = S.save(fig, OUT, "S2_faithfulness")
P.write(outs)
print(json.dumps(P.d["numbers"]))
