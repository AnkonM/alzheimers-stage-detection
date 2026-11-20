"""Does the gradient behind a layer1 CAM depend on the parity of the layer1 cell index? (LYP/.venv-mri, repo root)

    python tests/diagnostics/lattice_gradient_parity.py VOL96.nii.gz [MODEL_ID]

One frozen fold model (default 0) and one scan. Prints the mean |d ALZ logit / d layer1 activation| for each of the 8 parity classes of
the (x, y, z) cell index, relative to the mean of the 8 classes. A strided entry into layer2 samples layer1 unevenly, which would
give a period of two cells (8 voxels); this checks that for one model and one scan. It does not prove the mechanism.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "analysis", "explainability"))
import p1_lib as L  # noqa: E402

vol = L.load_volumes([sys.argv[1]])
model = L.get_model(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
acts = {}
hook = model.trunk.layer1.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
x = torch.from_numpy(vol).unsqueeze(1).to(L.DEVICE).requires_grad_(True)
g = torch.autograd.grad(model(x)["alz_logit"].sum(), acts["a"])[0][0]
hook.remove()
e = g.abs().mean(0).detach().cpu().numpy()
par = np.indices(e.shape) % 2
cls = {(a, b, c): float(e[(par[0] == a) & (par[1] == b) & (par[2] == c)].mean()) for a in (0, 1) for b in (0, 1) for c in (0, 1)}
mean = np.mean(list(cls.values()))
print(f"layer1 output grid {e.shape}, stride {vol.shape[-1] // e.shape[0]} voxels")
for k, v in cls.items():
    print(f"  parity {k}: {v / mean:.2f}")
