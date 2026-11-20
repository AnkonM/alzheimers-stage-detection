"""Supplementary S1: brain-age sanity on the training split (re-plot of F0010, panels 1 and 2).

Inputs: analysis/brain_age/brainage_trainval.csv (split == train, brain age available);
        analysis/brain_age/p2_bias_fits.csv (reference CN bias fit on all training CN: eval == "val" rows).
(a) DeepBrainNet predicted vs chronological age with the CN bias fit; (b) corrected brain-age gap by diagnosis.
The fit uses the formula of analysis/brain_age/p2_ablation.py fit_bias (OLS on CN rows) and must equal the
reference to 1e-6; the corrected gap uses p2_ablation.bag: (pred - beta) / alpha - age.
Run with LYP/.venv-mri.
"""
import io
import json

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from _prov import OUT, Prov
import paper_style as S

P = Prov("S1_brainage", __file__)
ba = pd.read_csv(io.BytesIO(P.file("analysis/brain_age/brainage_trainval.csv")))
ref = pd.read_csv(io.BytesIO(P.file("analysis/brain_age/p2_bias_fits.csv")))
src = P.file("analysis/brain_age/p2_ablation.py").decode()
P.check("fit_bias formula in p2_ablation.py is OLS of pred_age on age_at_scan over CN rows",
        "cn = df[(df.label == 0) & df.pred_age.notna()]" in src and "np.polyfit(cn.age_at_scan, cn.pred_age, 1)" in src, True, True)
P.check("bag formula in p2_ablation.py", "(df.pred_age - beta) / alpha - df.age_at_scan" in src, True, True)

tr = ba[(ba.split == "train") & ba.pred_age.notna()].copy()
cn = tr[tr.label == 0]
alpha, beta = np.polyfit(cn.age_at_scan, cn.pred_age, 1)
r = ref[ref["eval"] == "val"].iloc[0]
P.check("n_cn == reference n_cn (133)", len(cn) == int(r.n_cn) == 133, int(r.n_cn), len(cn))
P.check("alpha == reference (1e-6)", abs(alpha - r.alpha) < 1e-6, float(r.alpha), float(alpha))
P.check("beta == reference (1e-6)", abs(beta - r.beta) < 1e-6, float(r.beta), float(beta))
P.check("fit line == 0.582*age + 29.97", round(alpha, 3) == 0.582 and round(beta, 2) == 29.97, "0.582, 29.97",
        f"{alpha:.3f}, {beta:.2f}")
tr["BAG"] = (tr.pred_age - beta) / alpha - tr.age_at_scan
groups = [tr[tr.label == k].BAG.to_numpy() for k in range(4)]
H, p = kruskal(*groups)
p_text = "p < 1e-15" if p < 1e-15 else f"p = {p:.1e}"
P.check("Kruskal-Wallis H reproduces p2_report.md (76.0)", round(H, 1) == 76.0, 76.0, round(float(H), 2))
P.d["numbers"] = {"n_train_with_brain_age": len(tr), "n_cn": len(cn), "alpha": round(float(alpha), 5),
                  "beta": round(float(beta), 4), "kruskal_H": round(float(H), 2), "kruskal_p": float(p),
                  "p_shown_as": p_text, "n_by_dx": [len(x) for x in groups]}

import matplotlib.pyplot as plt  # noqa: E402

NAMES = ["CN", "EMCI", "LMCI", "AD"]
COL = [S.OKABE_ITO[c] for c in ("blue", "skyblue", "orange", "vermillion")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.17, wspace=0.28)
for k in range(4):
    d = tr[tr.label == k]
    a1.scatter(d.age_at_scan, d.pred_age, s=5, color=COL[k], alpha=0.75, lw=0, label=NAMES[k])
lim = np.array([54, 93])
a1.plot(lim, lim, color="0.5", ls="--", lw=0.8, label="Identity")
a1.plot(lim, alpha * lim + beta, color="black", lw=1.0, label=f"CN fit: {alpha:.3f}·age + {beta:.2f}")
a1.set(xlabel="Chronological age (years)", ylabel="Predicted brain age (years)", xlim=lim, ylim=(54, 99))
a1.legend(loc="upper left", ncol=2, markerscale=2, handlelength=1.2, columnspacing=0.8)
bp = a2.boxplot(groups, showfliers=False, widths=0.55, patch_artist=True, medianprops={"color": "black", "lw": 1})
for b, c in zip(bp["boxes"], COL):
    b.set(facecolor=c, alpha=0.6, lw=0.6)
a2.set_xticks(range(1, 5), NAMES)
a2.axhline(0, color="0.5", ls="--", lw=0.8)
a2.set_ylabel("Corrected brain-age gap (years)")
a2.text(0.03, 0.97, f"Kruskal–Wallis {p_text}", transform=a2.transAxes, va="top", ha="left")
S.panel_label(a1, "a", x=-0.12, y=1.04)
S.panel_label(a2, "b", x=-0.14, y=1.04)
P.check("p is never printed as 2.2e-16", "2.2e-16" not in p_text, True, p_text)
outs = S.save(fig, OUT, "S1_brainage")
P.write(outs)
print(json.dumps(P.d["numbers"]))
