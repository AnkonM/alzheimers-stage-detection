"""Diagnostic for the L1 parity ALZ failure (run with LYP/.venv-mri from the repo root).

    python tests/parity/diagnostics/alz_batch_diag.py <comma-separated RIDs>

Scores the stored test vol96 volumes with the frozen CNNs (analysis/temporal_eval/scan_alz.py loaders) in four
ways and prints max |ALZ - scan_inputs_test.csv ALZ| on the given subjects: all 123 in file order (the frozen
batching, chunks of 16), the subjects alone as one list, the same with deterministic cuDNN, and on the CPU.
Reads RID, PTID and ALZ only; no labels, no metrics.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "analysis", "temporal_eval"))
import scan_alz as SA  # noqa: E402
from predict_alz import load_model, score  # noqa: E402

sel = [int(x) for x in sys.argv[1].split(",")]
ref = pd.read_csv("analysis/fusion_v3/scan_inputs_test.csv", usecols=["RID", "PTID", "ALZ"]).dropna(subset=["ALZ"])
prep = "dataset/mri_prep/vol96"


def run(rows, device):
    vols = torch.stack([SA.load(f"{prep}/{p}.nii.gz") for p in rows.PTID])
    fm = np.empty((len(rows), 5))
    for k in range(5):
        fm[:, k] = np.mean([score(load_model(f"{SA.RUNS}/fold{k}_s{s}/best.pt", device)[0], vols, device)[1]
                            for s in (42, 43, 44)], 0)
    return fm.mean(1)


dev = torch.device("cuda")
pos = [int(np.flatnonzero(ref.RID.values == r)[0]) for r in sel]
sub = ref.iloc[pos]
print("all-123 (frozen batching) vs ref, on the subjects:", f"{np.abs(run(ref, dev)[pos] - sub.ALZ.values).max():.2e}")
print("subjects alone, GPU:", f"{np.abs(run(sub, dev) - sub.ALZ.values).max():.2e}")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("subjects alone, GPU deterministic cuDNN:", f"{np.abs(run(sub, dev) - sub.ALZ.values).max():.2e}")
print("subjects alone, CPU:", f"{np.abs(run(sub, torch.device('cpu')) - sub.ALZ.values).max():.2e}")
