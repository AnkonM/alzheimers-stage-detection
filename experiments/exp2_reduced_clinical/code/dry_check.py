"""Phase 2 dry-run checks (LYP/.venv-fusion). Reads only experiment outputs and, read-only,
the paper's streams.npz / kan_runs/{32,34}.npz for the Full-paper code check.

  python dry_check.py T1 42 0          shapes, NaNs, excluded columns, assertion counts
  python dry_check.py Full-paper 42 0  also: max |diff| vs the paper's saved predictions
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

tier, seed, fold = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
si = C.SEEDS.index(seed)
st = np.load(os.path.join(C.OOF, f"dry_streams_{tier}.npz"))
kn = np.load(os.path.join(C.OOF, f"dry_kan_{tier}.npz"))
meta = json.load(open(os.path.join(C.OOF, f"dry_streams_{tier}_meta.json")))
b, a = st[f"idx_test_{seed}_{fold}"], st[f"idx_train_{seed}_{fold}"]
rep = {"n_outer_train": len(a), "n_outer_test": len(b), "overlap": len(set(a) & set(b))}
for k in ("clin_oof", "fcb_oof"):
    rep[k] = {"shape": st[k].shape, "nan_in_fold": int(np.isnan(st[k][si, b]).sum()),
              "rows_sum_to_1": bool(np.allclose(st[k][si, b].sum(1), 1))}
rep["clin_inner"] = {"shape": st[f"clin_inner_{seed}_{fold}"].shape,
                     "nan": int(np.isnan(st[f"clin_inner_{seed}_{fold}"]).sum())}
rep["extras"] = {"names": meta[f"{seed}_{fold}"]["extras"], "tr": st[f"extras_tr_{seed}_{fold}"].shape,
                 "te": st[f"extras_te_{seed}_{fold}"].shape}
rep["top"] = meta[f"{seed}_{fold}"]["top"]
rep["indicators"] = meta[f"{seed}_{fold}"]["indicators"]
bad = set(rep["top"] + rep["extras"]["names"]) & C.EXCLUDED
rep["excluded_in_design"] = sorted(bad) if tier in C.REDUCED else "n/a (Full tier)"
for k in kn.files:
    if k.endswith("__cv"):
        rep[k] = {"nan_in_fold": int(np.isnan(kn[k][si, b]).sum()), "rows_sum_to_1": bool(np.allclose(kn[k][si, b].sum(1), 1))}
y = C.load()["train"][0].label.to_numpy()[b]
rep["fold_acc"] = {k: float((v[si, b].argmax(1) == y).mean()) for k, v in
                   [("C", st["clin_oof"]), ("F-CB", st["fcb_oof"])] + [(k.split("__")[0], kn[k]) for k in kn.files if k.endswith("__cv")]}

if tier == "Full-paper":
    old = np.load(os.path.join(C.V3, "streams.npz"))
    k32, k34 = np.load(os.path.join(C.V3, "kan_runs", "32.npz")), np.load(os.path.join(C.V3, "kan_runs", "34.npz"))
    rep["vs_paper"] = {
        "clin_oof": float(np.abs(st["clin_oof"][si, b] - old["clin_oof"][si, b]).max()),
        "clin_inner": float(np.abs(st[f"clin_inner_{seed}_{fold}"] - old[f"clin_inner_{seed}_{fold}"]).max()),
        "idx_test_equal": bool((b == old[f"idx_test_{seed}_{fold}"]).all()),
        "F-KAN_vs_32": float(np.abs(kn["F-KAN__cv"][si, b] - k32["cv"][si, b]).max()),
        "F-KAN-noMRI_vs_34": float(np.abs(kn["F-KAN-noMRI__cv"][si, b] - k34["cv"][si, b]).max()),
    }
cnt = collections.Counter(l.split("\t")[1] for l in open(os.path.join(C.LOGS, "assertions.log")))
rep["assertions_so_far"] = dict(cnt)
print(json.dumps(rep, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o)))
