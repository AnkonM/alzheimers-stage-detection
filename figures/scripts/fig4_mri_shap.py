"""Figure 4: MRI-specialist global SHAP for the AD centred score, training split (re-plot of F0008).

Input: analysis/explainability/results/p2_global_train.csv (class_target == AD; bars = mean_abs_phi, error bars =
bootstrap ci_lo/ci_hi). Labels from analysis/explainability/mri_shap_lib.py LABELS.
Also reports stream shares (volumes / ALZ + brain-age gap / age + sex) as % of total mean |SHAP| and compares
them with the committed analysis/explainability/results/p2_vs_controls.csv and the paper's 66 / 27 / 6.
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

P = Prov("fig4_mri_shap", __file__)
g = pd.read_csv(io.BytesIO(P.file("analysis/explainability/results/p2_global_train.csv")))
ref = pd.read_csv(io.BytesIO(P.file("analysis/explainability/results/p2_vs_controls.csv"))).set_index("group")
src = P.file("analysis/explainability/mri_shap_lib.py").decode()
node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LABELS")
LABELS = ast.literal_eval(node.value)
a = g[g.class_target == "AD"].copy()
P.check("n == 567 training subjects", (a.n == 567).all(), 567, a.n.unique().tolist())
P.check("10 inputs", len(a) == 10, 10, len(a))

GROUPS = {"volumes": ["Hippocampus_ICV", "Ventricles_ICV", "WholeBrain_ICV", "Entorhinal_ICV", "Fusiform_ICV", "MidTemp_ICV"],
          "ALZ+BAG": ["ALZ", "BAG"], "age+sex": ["AGE", "SEX"]}
P.check("groups cover every input once", sorted(sum(GROUPS.values(), [])) == sorted(a.feature), sorted(a.feature),
        sorted(sum(GROUPS.values(), [])))
tot = a.mean_abs_phi.sum()
share = {k: float(a[a.feature.isin(v)].mean_abs_phi.sum() / tot) for k, v in GROUPS.items()}
for k in GROUPS:
    P.check(f"share {k} == p2_vs_controls.csv", abs(share[k] - ref.loc[k, "share"]) < 1e-9, float(ref.loc[k, "share"]), share[k])
pct = {k: round(100 * v) for k, v in share.items()}
P.check("paper's 66 / 27 / 6 reproduce (rounded %)", [pct["volumes"], pct["ALZ+BAG"], pct["age+sex"]] == [66, 27, 6],
        [66, 27, 6], [pct["volumes"], pct["ALZ+BAG"], pct["age+sex"]])
allc = g.groupby("feature").mean_abs_phi.sum()
pooled = {k: round(100 * float(allc[v].sum() / allc.sum()), 1) for k, v in GROUPS.items()}
P.d["numbers"] = {"shares_AD_pct": {k: round(100 * v, 2) for k, v in share.items()},
                  "shares_all_classes_pooled_pct (info only)": pooled}

import matplotlib.pyplot as plt  # noqa: E402

a = a.sort_values("mean_abs_phi")
fig, ax = plt.subplots(figsize=(S.SINGLE, 3.1))
fig.subplots_adjust(left=0.40, right=0.97, top=0.80, bottom=0.14)
col = {f: S.OKABE_ITO[c] for c, fs in (("blue", GROUPS["volumes"]), ("orange", GROUPS["ALZ+BAG"]),
                                         ("skyblue", GROUPS["age+sex"])) for f in fs}
bars = ax.barh([LABELS[f] for f in a.feature], a.mean_abs_phi, color=[col[f] for f in a.feature], height=0.65,
               xerr=[a.mean_abs_phi - a.ci_lo, a.ci_hi - a.mean_abs_phi], error_kw={"lw": 0.7, "capsize": 1.5})
ax.set_xlabel("Mean |SHAP|, AD score (train, n = 567)")
from matplotlib.patches import Patch  # noqa: E402
ax.legend([Patch(color=S.OKABE_ITO[c]) for c in ("blue", "orange", "skyblue")],
          [f"Volumes ({pct['volumes']}%)", f"ALZ + brain-age gap ({pct['ALZ+BAG']}%)", f"Age + sex ({pct['age+sex']}%)"],
          loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncol=1, handlelength=1.0,
          borderaxespad=0.2)
P.check("plotted bar lengths == CSV", np.allclose([b.get_width() for b in bars], a.mean_abs_phi.to_numpy(), atol=0, rtol=0),
        a.mean_abs_phi.round(4).tolist(), [round(b.get_width(), 4) for b in bars])
outs = S.save(fig, OUT, "fig4_mri_shap")
P.write(outs)
print(json.dumps(P.d["numbers"]))
