"""Frozen-system core: cases -> ALZ, reliability, brain age, streams, KAN. Batch-native; one case = a batch of 1.

A case is a dict:
  key          str, unique in the batch
  raw          {feature: value} for the 35 step-2 source features (clinical + FreeSurfer-style volumes, ICV);
               missing ones are left out and imputed by the frozen rules inside the streams worker
  vol96, mni   m1 outputs (paths) or None when the case has no usable scan
  brain_age    bool: run DeepBrainNet (False when corr_vs_mni < 0.15, the frozen exclusion)
  age_at_scan  years
"""
import csv
import math
import os
import time

import numpy as np

from .clinical import stats


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([["" if v is None or (isinstance(v, float) and math.isnan(v)) else v for v in r] for r in rows])


def run_core(pool, cases, job_dir, explain=False, layercam=False):
    os.makedirs(job_dir, exist_ok=True)
    timings = {}

    def timed(name, fn, *a, **k):
        t = time.perf_counter()
        r = fn(*a, **k)
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - t
        return r
    s = stats()
    feats = s["num"] + s["cat"]
    keys = [c["key"] for c in cases]
    assert len(set(keys)) == len(keys)
    n = len(cases)
    alz, unrel, pred_age = [None] * n, [None] * n, [None] * n
    mri_rows = [i for i, c in enumerate(cases) if c.get("vol96")]
    if mri_rows:
        r = timed("alz", pool.mri.call, "alz", vol96_paths=[cases[i]["vol96"] for i in mri_rows])
        rl = timed("reliability", pool.mri.call, "reliability", mni_paths=[cases[i]["mni"] for i in mri_rows])
        for j, i in enumerate(mri_rows):
            alz[i], unrel[i] = r["alz"][j], rl["unreliable"][j]
    ba_rows = [i for i in mri_rows if cases[i].get("brain_age", True)]
    if ba_rows:
        b = timed("brain_age", pool.dbn.call, "brain_age", mni_paths=[cases[i]["mni"] for i in ba_rows])
        for j, i in enumerate(ba_rows):
            pred_age[i] = b["pred_age"][j]
    raw_csv, scan_csv = os.path.join(job_dir, "raw.csv"), os.path.join(job_dir, "scan.csv")
    _write_csv(raw_csv, ["key"] + feats, [[c["key"]] + [c["raw"].get(f) for f in feats] for c in cases])
    _write_csv(scan_csv, ["key", "ALZ", "unreliable", "pred_age", "age_at_scan"],
               [[c["key"], alz[i], unrel[i], pred_age[i], c.get("age_at_scan")] for i, c in enumerate(cases)])
    st = os.path.join(job_dir, "streams.npz")
    timed("streams_and_shap", pool.streams.call, "streams", raw_csv=raw_csv, scan_csv=scan_csv, out_npz=st)
    kn = os.path.join(job_dir, "kan.npz")
    timed("kan", pool.kan.call, "kan", streams_npz=st, out_npz=kn)
    k = np.load(kn)
    assert list(k["key"]) == [str(x) for x in keys]
    out = {}
    if explain:
        sh = os.path.join(job_dir, "shapley.npz")
        timed("kan_shapley", pool.kan.call, "shapley", streams_npz=st, out_npz=sh)
        out["shapley_npz"] = sh
    if layercam:
        cams = {}
        for i in mri_rows:
            cams[keys[i]] = timed("layercam", pool.mri.call, "layercam", vol96_path=cases[i]["vol96"],
                                  out_dir=os.path.join(job_dir, f"cam_{keys[i]}"))
        out["layercam"] = cams
    return {**out, "timings": timings, "key": keys, "alz": alz, "unreliable": unrel, "pred_age": pred_age,
            "streams_npz": st, "kan_npz": kn, "p": k["p"], "add": k["add"]}


def severity(p):
    return np.asarray(p) @ np.arange(4)
