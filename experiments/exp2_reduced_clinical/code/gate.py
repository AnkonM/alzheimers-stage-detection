"""Reproduction gate (LYP/.venv-fusion). Full-paper C and F-KAN CV macro-F1 vs the paper
(0.744, 0.756), tolerance ±0.010, under both definitions (§7: seed-averaged probabilities;
paper-style: mean of per-seed macro-F1), from the saved paper OOF (read-only) and from
this experiment's code. Writes logs/gate.json and tables/reproduction_gate.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import stats as ST  # noqa: E402

PAPER = {"C": 0.744, "F-KAN": 0.756}
TOL = 0.010

y = C.load()["train"][0].label.to_numpy()
old = np.load(os.path.join(C.V3, "streams.npz"))
k32 = np.load(os.path.join(C.V3, "kan_runs", "32.npz"))
st = np.load(os.path.join(C.OOF, "streams_Full-paper.npz"))
kn = np.load(os.path.join(C.OOF, "kan_Full-paper.npz"))
src = {"saved paper OOF": {"C": old["clin_oof"], "F-KAN": k32["cv"]},
       "experiment code": {"C": st["clin_oof"], "F-KAN": kn["F-KAN__cv"]}}
rows = []
for s, P in src.items():
    for sysname, A in P.items():
        assert not np.isnan(A).any()
        sec7 = ST.metrics_labels(y, A.mean(0).argmax(1))["macro_f1"]
        paper = float(np.mean([ST.metrics_labels(y, a.argmax(1))["macro_f1"] for a in A]))
        for d, v in (("spec §7 (seed-averaged probabilities)", sec7), ("paper-style (mean of per-seed)", paper)):
            rows.append({"source": s, "system": sysname, "definition": d, "paper": PAPER[sysname],
                         "reproduced": round(v, 4), "delta": round(v - PAPER[sysname], 4),
                         "pass": abs(v - PAPER[sysname]) <= TOL})
diff = {"C": float(np.abs(st["clin_oof"] - old["clin_oof"]).max()),
        "F-KAN": float(np.abs(kn["F-KAN__cv"] - k32["cv"]).max()),
        "F-KAN val": float(np.abs(kn["F-KAN__val"] - k32["val"]).max()),
        "C val": float(np.abs(st["clin_val"] - old["clin_val"]).max())}
df = pd.DataFrame(rows)
os.makedirs(C.TABLES, exist_ok=True)
df.to_csv(os.path.join(C.TABLES, "reproduction_gate.csv"), index=False)
res = {"pass": bool(df["pass"].all()), "rows": rows, "max_abs_diff_experiment_vs_saved": diff}
C.write_json(res, os.path.join(C.LOGS, "gate.json"))
print(df.to_string(index=False))
print("max |diff| experiment vs saved:", diff)
print("GATE", "PASS" if res["pass"] else "FAIL")
