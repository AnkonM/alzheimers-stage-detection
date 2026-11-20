"""KAN worker (LYP/.venv-kan): the frozen K2 residual KAN fusion, mean of 25 models.

Mirrors analysis/temporal_eval/infer_kan.py / fusion_v3/test_kan.py, with the 25 KANs and their input
transforms loaded once. Output: p (n, 4) stage probabilities, add (n,) AD-head probability.
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FV3 = os.path.join(ROOT, "analysis", "fusion_v3")
sys.path.insert(0, FV3)
sys.path.insert(0, os.path.join(ROOT, "analysis", "explainability"))
import common as XC  # noqa: E402
import kan_shap_lib as KS  # noqa: E402
import kan_fusion as KF  # noqa: E402
import metrics as M  # noqa: E402

from src.assessment.workers.client import serve  # noqa: E402

FROZEN = os.path.join(FV3, "frozen")
_K = {}


def load():
    if not _K:
        cfg = json.load(open(os.path.join(FROZEN, "kan_config.json")))
        _K["cfg"] = cfg
        for s in M.SEEDS:
            for ks in KF.KSEEDS:
                m = KF.build(cfg, 9 + len(cfg["extras"]), ks)
                m.load_state_dict(torch.load(os.path.join(FROZEN, f"kan_{s}_{ks}.pt")))
                _K[s, ks] = (m, pickle.load(open(os.path.join(FROZEN, f"kan_inputs_{s}_{ks}.pkl"), "rb")))
    return _K


def kan(streams_npz, out_npz):
    k = load()
    cfg = k["cfg"]
    z = np.load(streams_npz)
    P, A = [], []
    for s in M.SEEDS:
        for ks in KF.KSEEDS:
            m, inp = k[s, ks]
            with torch.no_grad():
                o = m(inp(z[f"clin_{s}"], z[f"mri_{s}"], z["has"], z["extras"]))
                P.append(torch.softmax(KF.cog_logits(cfg, o, inp.base(z[f"clin_{s}"])), 1).numpy())
                A.append(torch.sigmoid(o[:, 4]).numpy())
    np.savez(out_npz, key=z["key"], p=np.mean(P, 0), add=np.mean(A, 0), p_each=np.array(P))
    return {"n": int(len(z["key"])), "npz": out_npz}



def background():
    """Prediction-time inputs of the train subjects (SHAP background), from the card's streams table. Local cache."""
    path = os.path.join(ROOT, "var", "cache", "kan_bg.npz")
    if not os.path.exists(path):
        z = np.load(os.path.join(ROOT, "demo", "card_data", "streams_all.npz"), allow_pickle=True)
        sp = XC.split_of()
        tr = np.array([sp[int(r)] == "train" for r in z["rids"]])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, has=z["has"][tr], extras=z["extras"][tr],
                 **{f"{k}_{s}": z[f"{k}_{s}"][tr] for k in ("clin", "mri") for s in M.SEEDS})
    return dict(np.load(path))


_BG = {}


def shapley(streams_npz, out_npz):
    """Group Shapley values of the 25 KANs and the MRI-masked prediction (analysis/explainability P3, P4-B)."""
    if not _BG:
        _BG.update(background())
    bg = _BG
    z = np.load(streams_npz)
    has, ex = z["has"].astype(bool), z["extras"].astype(float)
    config = KS.cfg()
    G, PM = [], []
    for s, k in KS.MODELS:
        m, inp = KS.load_frozen(s, k, config)
        T, f, out, base = KS.terms(m, inp, z[f"clin_{s}"], z[f"mri_{s}"], has, ex)
        Tb, _, _, _ = KS.terms(m, inp, bg[f"clin_{s}"], bg[f"mri_{s}"], bg["has"].astype(bool), bg["extras"].astype(float))
        phi, ebg = KS.shapley(T, Tb)
        assert np.abs(phi.sum(1) + ebg - f).max() < 1e-4, "efficiency"
        G.append(KS.group(phi))
        with torch.no_grad():
            o2 = m(inp(z[f"clin_{s}"], z[f"mri_{s}"], np.zeros_like(has), ex)).numpy()
        PM.append(KS.softmax(o2[:, :4] + base.astype(np.float64)))
    np.savez(out_npz, phi_group=np.array(G, np.float32), p_mri_masked=np.mean(PM, 0).astype(np.float32),
             groups=np.array(KS.GROUP_NAMES), targets=np.array(KS.TARGETS))
    return {"n": int(len(has)), "npz": out_npz}

if __name__ == "__main__":
    serve({"kan": kan, "shapley": shapley, "ping": lambda: "kan"})
