"""B3: learned edge functions of the FROZEN 25-model KAN ensemble (no refitting). LYP/.venv-kan.

Models: frozen/kan_{s}_{ks}.pt + kan_inputs_{s}_{ks}.pkl, rebuilt with kan_fusion.build and
frozen/kan_config.json exactly as test_kan.py does. Fitting rows: the 691 train+val subjects in
frozen/streams_final.npz (outer seed s's stream scores feed KAN (s, ks)).
Gates (prereg B3):
  (a) reloaded models reproduce frozen_run/val_kan.npz p_each within 1e-5;
  (b) on the 691 fitting rows, node_scale * (subnode_scale * sum_i phi_ij(x_i) + subnode_bias) + node_bias,
      with phi_ij = pykan per-edge post-activations (mask included), reproduces model(x) within 1e-5.
Curves: 101 points per input over [min, max] of the raw input across the 691 fitting rows (pooled over the
5 outer seeds' stream scores; MRI logits over rows with a scan), mapped through each model's own scaler;
has_mri at {0, 1}. For every edge: phi (edge function) and its contribution to the output logit
(node_scale * subnode_scale * phi). Median and 10th/90th percentiles across the 25 models.
Outputs 0-3 are added to log p_clinical (residual COG logits); output 4 is the ADD logit.
"""
import csv
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

sys.path.insert(0, C.V3)
import kan_fusion as KF  # noqa: E402

FROZEN = os.path.join(C.V3, "frozen")
OUTS = ["COG_CN", "COG_EMCI", "COG_LMCI", "COG_AD", "ADD"]
NPTS = 101


def input_names(cfg):
    return ([f"clin_logit_{n}" for n in C.NAMES] + [f"mri_logit_{n}" for n in C.NAMES] + ["has_mri"]
            + cfg["extras"])


def load(pv, cfg, s, ks):
    m = KF.build(cfg, 9 + len(cfg["extras"]), ks)
    m.load_state_dict(torch.load(pv.file(os.path.join(FROZEN, f"kan_{s}_{ks}.pt"))))
    inp = pickle.load(open(pv.file(os.path.join(FROZEN, f"kan_inputs_{s}_{ks}.pkl")), "rb"))
    m.eval()
    return m, inp


def edges(m, X):
    """Per-edge post-activations (n, out, in) and the model output, one forward pass."""
    with torch.no_grad():
        o = m(X)
    return m.spline_postacts[0].numpy().astype(np.float64), o.numpy().astype(np.float64)


def main():
    torch.set_num_threads(4)
    pv = C.Prov("B3_kan_curves", __file__)
    cfg = json.load(open(pv.file(os.path.join(FROZEN, "kan_config.json"))))
    pv.check("frozen config is K2 width [13, 5], grid 5", cfg["variant"] == "K2" and cfg["G"] == 5
             and 9 + len(cfg["extras"]) == 13, observed=cfg)
    z = np.load(pv.file(os.path.join(FROZEN, "streams_final.npz")))
    vs = np.load(pv.file(os.path.join(C.V3, "frozen_run", "val_streams.npz")))
    vk = np.load(pv.file(os.path.join(C.V3, "frozen_run", "val_kan.npz")))
    C.assert_no_test(z["rids"], "streams_final.npz")
    C.assert_no_test(vs["rids"] if "rids" in vs.files else z["rids"][:0], "val_streams.npz")
    pv.check("streams_final has the 691 train+val rows", len(z["rids"]) == 691
             and set(z["rids"].astype(int)) == (C.TRAIN_RIDS | C.VAL_RIDS), 691, len(z["rids"]))
    has, ex = z["has"].astype(bool), z["extras"]
    names = input_names(cfg)
    fs = json.load(open(pv.file(os.path.join(FROZEN, "feature_stats.json"))))

    models, max_a, max_b = [], 0.0, 0.0
    raws = []
    for k, (s, ks) in enumerate([(s, ks) for s in C.SEEDS for ks in KF.KSEEDS]):
        m, inp = load(pv, cfg, s, ks)
        # gate (a): val predictions
        with torch.no_grad():
            o = m(inp(vs[f"clin_{s}"], vs[f"mri_{s}"], vs["has"], vs["extras"]))
            p = torch.softmax(KF.cog_logits(cfg, o, inp.base(vs[f"clin_{s}"])), 1).numpy()
        max_a = max(max_a, float(np.abs(p - vk["p_each"][k]).max()))
        # gate (b): additive decomposition on the 691 fitting rows
        X = inp(z[f"clin_oof_{s}"], z[f"mri_oof_{s}"], has, ex)
        post, out = edges(m, X)
        sub_s, sub_b = m.subnode_scale[0].detach().numpy(), m.subnode_bias[0].detach().numpy()
        nod_s, nod_b = m.node_scale[0].detach().numpy(), m.node_bias[0].detach().numpy()
        rec = nod_s * (sub_s * post.sum(2) + sub_b) + nod_b
        max_b = max(max_b, float(np.abs(rec - out).max()))
        raws.append(inp.raw(z[f"clin_oof_{s}"], z[f"mri_oof_{s}"], has, ex))
        models.append((s, ks, m, inp, nod_s * sub_s, m.act_fun[0].mask.detach().numpy().copy()))
    pv.check("gate (a): reloaded frozen KANs reproduce val_kan.npz p_each (max |diff| < 1e-5)", max_a < 1e-5,
             "< 1e-5", max_a)
    pv.check("gate (b): sum of per-edge activations + node affine reproduces model(x) on 691 rows (< 1e-5)",
             max_b < 1e-5, "< 1e-5", max_b)

    R = np.stack(raws)                                                    # (25, 691, 13) raw inputs
    rows = []
    curves = {}
    pruned = {}
    for i, nm in enumerate(names):
        if nm == "has_mri":
            xs = np.array([0.0, 1.0])
        else:
            sel = R[:, has, i] if nm.startswith("mri") else R[:, :, i]
            xs = np.linspace(sel.min(), sel.max(), NPTS)
        phi = np.empty((len(models), len(xs), 5))
        con = np.empty_like(phi)
        for k, (s, ks, m, inp, scale, mask) in enumerate(models):
            Xg = np.zeros((len(xs), len(names)))
            Xg[:, i] = xs if nm == "has_mri" else (xs - inp.mu[i]) / inp.sd[i]
            post, _ = edges(m, torch.tensor(Xg, dtype=torch.float32))
            phi[k] = post[:, :, i]
            con[k] = post[:, :, i] * scale[None, :]
            pruned.setdefault(nm, []).append(mask[i].tolist())
        curves[nm] = (xs, phi, con)
        orig = None
        if nm in ("AGE", "CDRSB", "MMSE"):
            orig = xs * fs["std"][nm] + fs["mean"][nm]
        for j, on in enumerate(OUTS):
            n_masked = int(sum(1 for mk in pruned[nm] if mk[j] == 0))
            for t, x in enumerate(xs):
                rows.append([nm, on, f"{x:.6g}", "" if orig is None else f"{orig[t]:.6g}",
                             *(f"{v:.6g}" for v in np.percentile(phi[:, t, j], [50, 10, 90])),
                             *(f"{v:.6g}" for v in np.percentile(con[:, t, j], [50, 10, 90])), n_masked])
    os.makedirs(os.path.join(C.OUT, "csv"), exist_ok=True)
    f = os.path.join(C.OUT, "csv", "B3_kan_edge_curves.csv")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["input", "output", "x_model_input_units", "x_original_units", "phi_median", "phi_p10", "phi_p90",
                    "logit_contribution_median", "logit_contribution_p10", "logit_contribution_p90",
                    "n_models_edge_masked_of_25"])
        w.writerows(rows)
    outs = [f]
    pv.note("x_model_input_units: clinical/MRI inputs are centred log-probabilities (no temperature); AGE, CDRSB, "
            "MMSE, CogIndex are the step-4 z-scored values of frozen/feature_stats.json (fit on 691); "
            "x_original_units = x * std + mean from feature_stats.json for AGE (years), CDRSB, MMSE (points). "
            "CogIndex has no original unit. Curves are not centred: each edge's offset is part of the model.")
    pv.note("logit_contribution = node_scale * subnode_scale * phi: the edge's additive term in output j. "
            "Outputs COG_* are added to log p_clinical (K2 residual); ADD is the AD-vs-rest logit.")
    outs += figure(curves, names)
    C.write_json({"gate_a_max_abs_diff": max_a, "gate_b_max_abs_diff": max_b,
                  "masked_edges": {nm: int(np.sum(np.array(v) == 0)) for nm, v in pruned.items()}},
                 os.path.join(C.OUT, "json", "B3.json"))
    pv.write(outs + [os.path.join(C.OUT, "json", "B3.json")])
    print("gate a", max_a, "gate b", max_b)


def figure(curves, names):
    sys.path.insert(0, C.HERE)
    import paper_style as PS
    plt = PS.plt
    fs = json.load(open(os.path.join(FROZEN, "feature_stats.json")))
    labels = {"has_mri": "has MRI (0/1)", "AGE": "Age (years)", "CDRSB": "CDR-SB", "MMSE": "MMSE",
              "CogIndex": "CogIndex (z units)"}
    for n in C.NAMES:
        labels[f"clin_logit_{n}"] = f"Clinical logit {n}"
        labels[f"mri_logit_{n}"] = f"MRI logit {n}"
    fig, axes = plt.subplots(len(names), 5, figsize=(PS.DOUBLE, 14.0), sharex="row")
    for i, nm in enumerate(names):
        xs, _, con = curves[nm]
        xp = xs * fs["std"][nm] + fs["mean"][nm] if nm in ("AGE", "CDRSB", "MMSE") else xs
        for j in range(5):
            ax = axes[i, j]
            med, lo, hi = (np.percentile(con[:, :, j], q, axis=0) for q in (50, 10, 90))
            if nm == "has_mri":
                ax.errorbar([0, 1], med, yerr=[med - lo, hi - med], fmt="o", ms=3, color=PS.OKABE_ITO["blue"],
                            capsize=2, lw=0.8)
                ax.set_xticks([0, 1])
                ax.set_xlim(-0.5, 1.5)
            else:
                ax.fill_between(xp, lo, hi, color=PS.OKABE_ITO["skyblue"], alpha=0.45, lw=0)
                ax.plot(xp, med, color=PS.OKABE_ITO["blue"], lw=1.0)
            ax.axhline(0, color=PS.OTHER_GREY, lw=0.5, zorder=0)
            ax.locator_params(axis="y", nbins=3)
            ax.locator_params(axis="x", nbins=2)
            if i == 0:
                ax.set_title(OUTS[j].replace("_", " "))
            if j == 0:
                ax.set_ylabel(labels[nm], rotation=0, ha="right", va="center")
    fig.supylabel("Contribution to output logit (median, 10–90% band over 25 frozen KANs)", x=0.005)
    fig.tight_layout(rect=(0.015, 0, 1, 1), h_pad=0.4, w_pad=0.6)
    return list(PS.save(fig, os.path.join(C.OUT, "figures"), "B3_kan_edge_curves"))


if __name__ == "__main__":
    main()
