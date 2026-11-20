"""Figure 3: LayerCAM (layer1) cascading model-randomisation sanity check (re-plot of F0007).

Input: analysis/explainability/results/p1_sanity_cascade.csv (committed run, n = 50 training subjects; not the
dry-run cache). The 0.5 line is the pre-registered PASS threshold (p1_sanity.verdict_cascade: PASS < 0.5).
Run with LYP/.venv-mri.
"""
import ast
import io
import json
import os

import numpy as np
import pandas as pd

from _prov import OUT, WT, Prov
import paper_style as S

P = Prov("fig3_cascade", __file__)
PATH = "analysis/explainability/results/p1_sanity_cascade.csv"
P.check("input is the committed results file, not the dry-run cache", "cache/dry" not in PATH, True, True)
t = pd.read_csv(io.BytesIO(P.file(PATH)))
src = open(os.path.join(WT, "analysis/explainability/p1_sanity.py")).read()
P.check("PASS threshold in p1_sanity.verdict_cascade is 0.5", 'return "PASS" if med < 0.5' in src, 0.5, 0.5 if 'med < 0.5' in src else None)
P.check("n == 50 training subjects at every step", (t.n == 50).all(), 50, t.n.unique().tolist())
steps = t.randomised_through.tolist()
P.check("step order", steps == ["alz_head", "layer4", "layer3", "layer2", "layer1", "stem"], None, steps)
labels = {"alz_head": "Head", "layer4": "+ layer 4", "layer3": "+ layer 3", "layer2": "+ layer 2",
          "layer1": "+ layer 1", "stem": "+ stem"}

import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(S.SINGLE, 2.45))
fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.27)
x = np.arange(len(t))
c = S.OKABE_ITO["blue"]
ax.fill_between(x, t.p25, t.p75, color=c, alpha=0.25, lw=0, label="IQR")
line, = ax.plot(x, t.median_spearman, marker="o", ms=3.5, color=c, label="Median")
ax.axhline(0.5, color="0.35", lw=0.8, ls="--")
ax.text(len(t) - 1, 0.47, "pass threshold", ha="right", va="top", fontsize=8, color="0.25")
ax.set_xticks(x, [labels[s] for s in steps], rotation=35, ha="right")
ax.set_ylabel("Spearman ρ with original map")
ax.set_ylim(0, 1.0)
ax.set_xlim(-0.3, len(t) - 0.7)
ax.legend(loc="lower left", ncol=2, handlelength=1.4)
P.check("plotted medians == CSV", np.array_equal(line.get_ydata(), t.median_spearman.to_numpy()),
        t.median_spearman.round(4).tolist(), np.round(line.get_ydata(), 4).tolist())
P.d["numbers"] = {"median": dict(zip(steps, t.median_spearman.round(4))), "p25": dict(zip(steps, t.p25.round(4))),
                  "p75": dict(zip(steps, t.p75.round(4)))}
outs = S.save(fig, OUT, "fig3_cascade")
P.write(outs)
print(json.dumps(P.d["numbers"]))
