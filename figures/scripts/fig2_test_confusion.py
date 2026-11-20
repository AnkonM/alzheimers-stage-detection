"""Figure 2: frozen fusion-v3 (KAN K2) confusion matrix on the held-out test split (n = 123).

Input: analysis/fusion_v3/frozen_run/test_results.json (stored confusion matrix; nothing is recomputed from
model outputs). Rows = true stage (row sums must equal the stored class counts), columns = predicted stage.
Per-class F1 and macro-F1 are recomputed from the matrix and must equal the stored values before plotting.
Run with LYP/.venv-mri.
"""
import json

import numpy as np

from _prov import OUT, Prov
import paper_style as S

P = Prov("fig2_test_confusion", __file__)
R = json.loads(P.file("analysis/fusion_v3/frozen_run/test_results.json"))
SYS = "KAN fusion (final, K2)"
NAMES = ["CN", "EMCI", "LMCI", "AD"]
P.check("per-cell confusion matrix present for the final system", SYS in R.get("confusion", {}), True, SYS in R.get("confusion", {}))
cm = np.array(R["confusion"][SYS], dtype=int)
P.check("matrix is 4x4", cm.shape == (4, 4), [4, 4], list(cm.shape))
P.check("n = 123", int(cm.sum()) == R["n"] == 123, 123, int(cm.sum()))
P.check("rows are true classes (row sums == stored labels)", cm.sum(1).tolist() == R["labels"], R["labels"], cm.sum(1).tolist())

tp = np.diag(cm).astype(float)
prec, rec = tp / cm.sum(0), tp / cm.sum(1)
f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
stored = R["systems"][SYS]
for k, n in enumerate(NAMES):
    P.check(f"F1 {n} from matrix == stored f1_{n}", abs(f1[k] - stored[f"f1_{n}"]) < 1e-9, stored[f"f1_{n}"], float(f1[k]))
P.check("macro-F1 from matrix == stored macro_f1", abs(f1.mean() - stored["macro_f1"]) < 1e-9, stored["macro_f1"], float(f1.mean()))
P.check("macro-F1 rounds to 0.764", round(float(f1.mean()), 3) == 0.764, 0.764, round(float(f1.mean()), 3))
acc = tp.sum() / cm.sum()
P.check("accuracy from matrix == stored acc", abs(acc - stored["acc"]) < 1e-9, stored["acc"], float(acc))

import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(S.SINGLE, 3.05))
fig.subplots_adjust(left=0.24, right=0.98, top=0.98, bottom=0.17)
ax.imshow(cm / cm.sum(1, keepdims=True), cmap=S.SEQ_CMAP, vmin=0, vmax=1)
drawn = {}
for i in range(4):
    for j in range(4):
        r = cm[i, j] / cm[i].sum()
        ax.text(j, i, f"{cm[i, j]}\n{100 * r:.0f}%", ha="center", va="center", fontsize=8,
                color="white" if r > 0.55 else "black", linespacing=1.1)
        drawn[(i, j)] = int(cm[i, j])
ax.set_xticks(range(4), NAMES)
ax.set_yticks(range(4), [f"{n}\nF1 {f1[k]:.2f}" for k, n in enumerate(NAMES)])
ax.set_xlabel("Predicted stage")
ax.set_ylabel("True stage")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
P.check("drawn counts == stored matrix", all(drawn[(i, j)] == cm[i, j] for i in range(4) for j in range(4)), True, True)
P.d["numbers"] = {"macro_f1": round(float(f1.mean()), 4), "f1": dict(zip(NAMES, np.round(f1, 4).tolist())),
                  "recall": dict(zip(NAMES, np.round(rec, 4).tolist())), "accuracy": round(float(acc), 4)}
outs = S.save(fig, OUT, "fig2_test_confusion")
P.write(outs)
print(json.dumps(P.d["numbers"]))
