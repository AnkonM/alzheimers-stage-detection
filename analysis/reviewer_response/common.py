"""Shared helpers for the reviewer-response analyses (docs/reviewer_response_prereg.md).

numpy / pandas / sklearn only, so it loads in LYP/.venv-fusion and LYP/.venv-kan.
- Paths: ROOT = this worktree; OUT = deliverable build dir outside the repo; PRIV = gitignored
  subject-level intermediates.
- Test guard: assert_no_test() aborts if a test RID is present (only b4_operating.py reads test
  predictions, through git blobs, and never calls a loader that would read test features).
- Bootstrap: 2,000 resamples, np.random.default_rng(0), the same indices for every comparison on
  the same subjects; statistic = per-seed metric averaged over seeds (evaluate.py convention).
- Prov: records inputs (sha256 / git blob), checks and notes into OUT/provenance/<step>.json; a
  failed check is written first, then raises.
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
V3 = os.path.join(ROOT, "analysis", "fusion_v3")
KF = os.path.join(ROOT, "analysis", "kan_fusion")
PRIV = os.path.join(HERE, "private")
OUT = os.environ.get("RR_OUT", os.path.join(os.path.dirname(os.path.dirname(ROOT)), "reviewer_response_2026-11-18",
                                            "build"))
SEEDS = (42, 43, 44, 45, 46)
NAMES = ["CN", "EMCI", "LMCI", "AD"]
N_BOOT = 2000

sys.path.insert(0, V3)
import metrics as M  # noqa: E402  (fusion_v3 metrics: macro_f1_boot, metrics, cv_summary; read-only use)

TEST_RIDS = frozenset(pd.read_csv(os.path.join(ROOT, "dataset", "processed", "test_rids.csv")).RID.astype(int))
TRAIN_RIDS = frozenset(pd.read_csv(os.path.join(ROOT, "dataset", "processed", "train_rids.csv")).RID.astype(int))
VAL_RIDS = frozenset(pd.read_csv(os.path.join(ROOT, "dataset", "processed", "val_rids.csv")).RID.astype(int))
assert len(TEST_RIDS) == 123 and len(TRAIN_RIDS) == 569 and len(VAL_RIDS) == 122
assert not (TEST_RIDS & (TRAIN_RIDS | VAL_RIDS))


def assert_no_test(rids, where):
    bad = set(np.asarray(rids).astype(int).tolist()) & TEST_RIDS
    assert not bad, f"test RIDs present in {where} ({len(bad)})"


def load_trainval():
    d = pd.read_csv(os.path.join(KF, "fusion_trainval.csv"))
    assert_no_test(d.RID, "fusion_trainval.csv")
    return d[d.split == "train"].reset_index(drop=True), d[d.split == "val"].reset_index(drop=True)


# --- bootstrap ---------------------------------------------------------------------------
def boot_idx(n):
    return np.random.default_rng(0).integers(0, n, size=(N_BOOT, n))


def f1_seeds(y, P):
    """P: (n_seeds, n, 4) -> per-seed macro-F1 array."""
    return np.array([M.metrics(y, p)["macro_f1"] for p in P])


def auc_seeds(y, P):
    return np.array([roc_auc_score(y == 3, p[:, 3]) for p in P])


def f1_boot(y, P, idx):
    """(B,) bootstrap statistic: macro-F1 per seed, mean over seeds."""
    return M.macro_f1_boot(y, P.argmax(2), idx).mean(1)


def auc_boot(y, P, idx):
    pos = (y == 3)
    out = np.empty(len(idx))
    for b, ii in enumerate(idx):
        yy = pos[ii]
        out[b] = np.mean([roc_auc_score(yy, p[ii, 3]) for p in P]) if 0 < yy.sum() < len(yy) else np.nan
    return out


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def fmt(x, d=3):
    return "—" if x is None else f"{x:+.{d}f}" if isinstance(x, float) and x < 0 else f"{x:.{d}f}"


def fmt_ci(c, d=3, signed=True):
    f = (lambda v: f"{v:+.{d}f}") if signed else (lambda v: f"{v:.{d}f}")
    return f"[{f(c[0])}, {f(c[1])}]"


# --- provenance --------------------------------------------------------------------------
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


class Prov:
    def __init__(self, step, script):
        rel = os.path.relpath(script, ROOT)
        self.t0 = time.time()
        self.d = {"step": step, "script": rel, "commit": git("rev-parse", "--short", "HEAD").decode().strip(),
                  "branch": git("rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
                  "script_committed_unmodified": subprocess.run(
                      ["git", "diff", "--quiet", "HEAD", "--", rel, "analysis/reviewer_response/common.py"],
                      cwd=ROOT).returncode == 0 and rel in git("ls-files", rel).decode(),
                  "python": sys.executable, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "inputs": [], "checks": [], "notes": []}

    def file(self, path):
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        b = open(full, "rb").read()
        self.d["inputs"].append({"path": os.path.relpath(full, ROOT), "sha256": sha256_bytes(b), "bytes": len(b)})
        return full

    def git_blob(self, ref, path):
        b = git("show", f"{ref}:{path}")
        self.d["inputs"].append({"path": f"{ref}:{path}", "git_blob": git("rev-parse", f"{ref}:{path}").decode().strip(),
                                 "sha256": sha256_bytes(b), "bytes": len(b)})
        return b

    def npz_blob(self, ref, path):
        return np.load(io.BytesIO(self.git_blob(ref, path)), allow_pickle=False)

    def check(self, name, ok, expected=None, observed=None):
        self.d["checks"].append({"name": name, "pass": bool(ok), "expected": _j(expected), "observed": _j(observed)})
        if not ok:
            self.write()
            raise AssertionError(f"[{self.d['step']}] check failed: {name}: expected {expected}, observed {observed}")

    def note(self, text):
        self.d["notes"].append(text)

    def write(self, outputs=()):
        self.d["outputs"] = [os.path.relpath(o, OUT) for o in outputs]
        self.d["runtime_s"] = round(time.time() - self.t0, 1)
        os.makedirs(os.path.join(OUT, "provenance"), exist_ok=True)
        with open(os.path.join(OUT, "provenance", f"{self.d['step']}.json"), "w") as f:
            json.dump(self.d, f, indent=1)


def _j(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return path
