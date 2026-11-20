"""Frozen KAN fusion per scan (run with LYP/.venv-kan). Inference only; mirrors analysis/fusion_v3/test_kan.py.

COG / ADD = mean over the 25 frozen KANs; KAN (s, k) takes the seed-s specialists' scores.
--gate      private/gate_test_streams.npz -> private/gate_test_kan.npz, compared with frozen_run/test_kan.npz (G1a).
--ingestion private/ingestion_streams.npz -> private/ingestion_kan.npz.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FV3 = os.path.join(ROOT, "analysis", "fusion_v3")
sys.path.insert(0, FV3)
import kan_fusion as KF  # noqa: E402
import metrics as M  # noqa: E402

FROZEN = os.path.join(FV3, "frozen")
PRIVATE = os.path.join(HERE, "private")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gate", action="store_true")
    g.add_argument("--ingestion", action="store_true")
    a = ap.parse_args()
    tag = "gate_test" if a.gate else "ingestion"
    cfg = json.load(open(os.path.join(FROZEN, "kan_config.json")))
    z = np.load(os.path.join(PRIVATE, f"{tag}_streams.npz"))
    P, A = [], []
    for s in M.SEEDS:
        for ks in KF.KSEEDS:
            m = KF.build(cfg, 9 + len(cfg["extras"]), ks)
            m.load_state_dict(torch.load(os.path.join(FROZEN, f"kan_{s}_{ks}.pt")))
            inp = pickle.load(open(os.path.join(FROZEN, f"kan_inputs_{s}_{ks}.pkl"), "rb"))
            with torch.no_grad():
                o = m(inp(z[f"clin_{s}"], z[f"mri_{s}"], z["has"], z["extras"]))
                P.append(torch.softmax(KF.cog_logits(cfg, o, inp.base(z[f"clin_{s}"])), 1).numpy())
                A.append(torch.sigmoid(o[:, 4]).numpy())
    p, add = np.mean(P, 0), np.mean(A, 0)
    np.savez(os.path.join(PRIVATE, f"{tag}_kan.npz"), key=z["key"], p=p, add=add)
    print(f"{tag}: KAN fusion applied ({len(P)} KANs) to {len(p)} rows")
    if a.gate:
        ref = np.load(os.path.join(FV3, "frozen_run", "test_kan.npz"))
        dp, da = float(np.abs(ref["p"] - p).max()), float(np.abs(ref["add"] - add).max())
        same = bool((ref["p"].argmax(1) == p.argmax(1)).all())
        print(f"G1a KAN: max |dp| {dp:.2e}, max |d add| {da:.2e}, identical argmax {same}, "
              f"pass {dp <= 1e-6 and da <= 1e-6 and same}")


if __name__ == "__main__":
    main()
