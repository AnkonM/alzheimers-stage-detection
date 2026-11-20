"""[reviewer-response B2 copy] Source: 880f4a7:experiments/exp2_reduced_clinical/code/kan_run.py (unchanged; paths via common.py).

EXP2 KAN fusion per tier (LYP/.venv-kan + PYTHONPATH=LYP/.pylib-exp2). Train/val only.

Uses analysis/fusion_v3/kan_fusion.fit_predict unchanged with the frozen config 32
(read from kan_runs/32.npz, read-only). Nothing is written under analysis/.
Systems:
  F-KAN           config 32 on clin (tier) + MRI stream scores + has_mri + tier extras.
  F-KAN-mask-inf  the F-KAN models applied with has_mri = 0 on the evaluation rows.
  F-KAN-noMRI     config 32 with no_mri = True (retrained; MRI masked in train and eval).
Units as the paper: CV outer fold f (KAN trains on inner OOF scores of the outer-train
set, predicts the outer-test fold) and "val" (trains on the 569 OOF scores, predicts val).
Output: oof/kan_<tier>.npz (or oof/dry_kan_<tier>.npz with --dry).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

sys.path.insert(0, C.V3)
import kan_fusion as K  # noqa: E402  (fit_predict, cog_logits only)

CFG = {**K.DEFAULT, **json.loads(str(np.load(os.path.join(C.V3, "kan_runs", "32.npz"))["cfg"]))}
G = {}


def init(tier, pre):
    torch.set_num_threads(1)
    data = C.load()
    G["tr"], G["va"] = data["train"][0], data["val"][0]
    G["st"] = dict(np.load(os.path.join(C.OOF, f"{pre}streams_{tier}.npz")))
    G["mri"] = dict(np.load(os.path.join(C.OOF, f"{pre}streams_MRI.npz")))
    for k in ("train_rids", "val_rids"):
        assert (G["st"][k] == G["mri"][k]).all()
    assert (G["st"]["train_rids"] == G["tr"].RID.to_numpy()).all()


def unit(seed, u):
    tr, va, st, mr = G["tr"], G["va"], G["st"], G["mri"]
    si = C.SEEDS.index(seed)
    if u == "val":
        a = np.arange(len(tr))
        trd = dict(pc=st["clin_oof"][si], pm=mr["mri_oof"][si], has=tr.has_mri.to_numpy(), y=tr.label.to_numpy(),
                   extras=st[f"extras_tr_{seed}_val"], rows=a)
        ted = dict(pc=st["clin_val"][si], pm=mr["mri_val"][si], has=va.has_mri.to_numpy(),
                   extras=st[f"extras_te_{seed}_val"], rows=np.arange(len(va)))
        C.assert_disjoint(tr.RID, va.RID, f"KAN s{seed} val")
    else:
        a, b = st[f"idx_train_{seed}_{u}"], st[f"idx_test_{seed}_{u}"]
        trd = dict(pc=st[f"clin_inner_{seed}_{u}"], pm=mr[f"mri_inner_{seed}_{u}"], has=tr.has_mri.to_numpy()[a],
                   y=tr.label.to_numpy()[a], extras=st[f"extras_tr_{seed}_{u}"], rows=a)
        ted = dict(pc=st["clin_oof"][si][b], pm=mr["mri_oof"][si][b], has=tr.has_mri.to_numpy()[b],
                   extras=st[f"extras_te_{seed}_{u}"], rows=b)
        C.assert_disjoint(tr.RID.iloc[a], tr.RID.iloc[b], f"KAN s{seed} f{u}")
    for d in (trd, ted):
        assert not np.isnan(d["pc"]).any() and not np.isnan(d["extras"]).any()
    return trd, ted


def job(arg):
    system, seed, u = arg
    t0 = time.time()
    trd, ted = unit(seed, u)
    cfg = dict(CFG)
    if system == "F-KAN-noMRI":
        cfg["no_mri"] = True
        trd["has"], ted["has"] = np.zeros_like(trd["has"]), np.zeros_like(ted["has"])
    res = {"p": [], "a": [], "p_mask": [], "a_mask": [], "fails": 0, "steps": []}
    for ks in K.KSEEDS:
        p, a, dg, (model, inp) = K.fit_predict(cfg, trd, ted, ks)
        res["p"].append(p)
        res["a"].append(a)
        res["fails"] += dg["failed"]
        res["steps"].append(dg["best_step"])
        if system == "F-KAN":
            with torch.no_grad():
                z = np.zeros_like(ted["has"])
                out = model(inp(ted["pc"], ted["pm"], z, ted["extras"]))
                res["p_mask"].append(torch.softmax(K.cog_logits(cfg, out, inp.base(ted["pc"])), 1).numpy())
                res["a_mask"].append(torch.sigmoid(out[:, 4]).numpy())
    return system, seed, u, ted["rows"], {k: (np.mean(v, 0) if k in ("p", "a", "p_mask", "a_mask") and v else v)
                                          for k, v in res.items()}, trd["extras"].shape[1] + 9, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS))
    ap.add_argument("--units", nargs="+", default=["0", "1", "2", "3", "4", "val"])
    ap.add_argument("--systems", nargs="+", default=["F-KAN", "F-KAN-noMRI"])
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    pre = "dry_" if a.dry else ""
    units = [int(u) if u != "val" else u for u in a.units]
    init(a.tier, pre)
    n_tr, n_va = len(G["tr"]), len(G["va"])
    keys = {"F-KAN": ["F-KAN", "F-KAN-mask-inf"], "F-KAN-noMRI": ["F-KAN-noMRI"]}
    out = {}
    for s in a.systems:
        for k in keys[s]:
            out[f"{k}|cv"], out[f"{k}|val"] = np.full((5, n_tr, 4), np.nan), np.full((5, n_va, 4), np.nan)
            out[f"{k}|cv_add"], out[f"{k}|val_add"] = np.full((5, n_tr), np.nan), np.full((5, n_va), np.nan)
    logf = open(os.path.join(C.LOGS, f"{pre}kan_{a.tier}.log"), "a")
    diag = []
    t0 = time.time()
    jobs = [(s, seed, u) for s in a.systems for seed in a.seeds for u in units]
    with Pool(min(a.workers, len(jobs)), initializer=init, initargs=(a.tier, pre)) as pool:
        for system, seed, u, rows, r, n_in, dt in pool.imap_unordered(job, jobs):
            si, ev = C.SEEDS.index(seed), ("val" if u == "val" else "cv")
            out[f"{system}|{ev}"][si, rows] = r["p"]
            out[f"{system}|{ev}_add"][si, rows] = r["a"]
            if system == "F-KAN":
                out[f"F-KAN-mask-inf|{ev}"][si, rows] = r["p_mask"]
                out[f"F-KAN-mask-inf|{ev}_add"][si, rows] = r["a_mask"]
            d = {"system": system, "seed": seed, "unit": str(u), "n_in": n_in, "fails": r["fails"],
                 "best_steps": r["steps"], "seconds": round(dt, 1)}
            diag.append(d)
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {a.tier} {json.dumps(d)}"
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()
    np.savez(os.path.join(C.OOF, f"{pre}kan_{a.tier}.npz"), **{k.replace("|", "__"): v for k, v in out.items()})
    C.write_json({"diag": diag, "runtime_s": time.time() - t0, "cfg": CFG},
                 os.path.join(C.OOF, f"{pre}kan_{a.tier}_meta.json"))
    print(f"{a.tier} KAN done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
