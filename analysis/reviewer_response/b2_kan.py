"""B2 step 2 (LYP/.venv-kan): KAN config 32 (final) and 34 (MRI masked) on the fold-wise stream scores.

kan_fusion.fit_predict is imported unchanged; configs are read from kan_runs/{32,34}.npz. For each outer
seed / fold the KAN trains on the fold-wise inner OOF scores with fold-wise extras (FoldPrep fitted on the
outer-train rows = the KAN's training rows) and predicts the outer-test fold, 5 KAN seeds averaged, as
kan_fusion.job does. Harness check first: the same code on the ORIGINAL streams.npz + fusion_trainval
extras reproduces kan_runs/32.npz on seed 42 fold 0.
Output: private/b2_kan_foldwise.npz (cv32, cv34).
"""
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
import kan_fusion as K  # noqa: E402

FW = os.path.join(C.PRIV, "b2_streams_foldwise.npz")
G = {}


def cfg_of(cid):
    return {**K.DEFAULT, **json.loads(str(np.load(os.path.join(C.V3, "kan_runs", f"{cid}.npz"))["cfg"]))}


def init():
    torch.set_num_threads(1)
    G["fw"] = dict(np.load(FW))


def run_unit(cfg, trd, ted):
    if cfg["no_mri"]:
        trd, ted = dict(trd, has=np.zeros_like(trd["has"])), dict(ted, has=np.zeros_like(ted["has"]))
    ps = [K.fit_predict(cfg, trd, ted, ks)[0] for ks in K.KSEEDS]
    return np.mean(ps, 0)


def job(arg):
    cid, seed, f = arg
    z = G["fw"]
    si = C.SEEDS.index(seed)
    a, b = z[f"idx_train_{seed}_{f}"], z[f"idx_test_{seed}_{f}"]
    trd = dict(pc=z[f"clin_inner_{seed}_{f}"], pm=z[f"mri_inner_{seed}_{f}"], has=z["has"][a], y=z["y"][a],
               extras=z[f"extras_tr_{seed}_{f}"], rows=a)
    ted = dict(pc=z["clin_oof"][si][b], pm=z["mri_oof"][si][b], has=z["has"][b], extras=z[f"extras_te_{seed}_{f}"],
               rows=b)
    assert not np.isnan(trd["extras"]).any() and not np.isnan(ted["extras"]).any()
    return cid, seed, b, run_unit(cfg_of(cid), trd, ted)


def main():
    torch.set_num_threads(4)
    pv = C.Prov("B2_kan", __file__)
    for p in (FW, os.path.join(C.V3, "kan_runs", "32.npz"), os.path.join(C.V3, "kan_runs", "34.npz"),
              os.path.join(C.V3, "streams.npz"), os.path.join(C.V3, "kan_fusion.py")):
        pv.file(p)
    # harness check on the original inputs
    K.init_worker()
    trd, ted, trdf, _ = K.unit_data(42, 0)
    cfg = cfg_of(32)
    for d in (trd, ted):
        d["extras"] = trdf.iloc[d["rows"]][cfg["extras"]].to_numpy(dtype=float)
    ref = np.load(os.path.join(C.V3, "kan_runs", "32.npz"))["cv"][0, ted["rows"]]
    dd = float(np.abs(run_unit(cfg, trd, ted) - ref).max())
    pv.check("harness check: original inputs reproduce kan_runs/32.npz on seed 42 fold 0 (< 1e-5)", dd < 1e-5,
             "< 1e-5", dd)
    z = np.load(FW)
    C.assert_no_test(z["train_rids"], "b2_streams_foldwise.npz")
    n = len(z["y"])
    P = {32: np.full((5, n, 4), np.nan), 34: np.full((5, n, 4), np.nan)}
    t0 = time.time()
    with Pool(15, initializer=init) as pool:
        for cid, seed, b, p in pool.imap_unordered(job, [(c, s, f) for c in (32, 34) for s in C.SEEDS
                                                         for f in range(5)]):
            P[cid][C.SEEDS.index(seed), b] = p
    pv.check("every subject predicted", all(np.isfinite(v).all() for v in P.values()))
    np.savez(os.path.join(C.PRIV, "b2_kan_foldwise.npz"), cv32=P[32], cv34=P[34])
    pv.note(f"KAN fits: 2 configs x 25 units x 5 KAN seeds in {time.time() - t0:.0f}s")
    pv.write()
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
