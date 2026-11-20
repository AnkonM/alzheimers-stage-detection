"""Brain-age worker (LYP/.venv-deepbrainnet, TF_USE_LEGACY_KERAS=1): DeepBrainNet on the m1 `mni` volume.

Reuses analysis/brain_age/dbn_smoke.py (to_dbn_grid, predict) unchanged; weights sha256-checked by the registry.
The corr_vs_mni < 0.15 exclusion is applied by the engine before calling.
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only, as in the frozen run

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "analysis", "brain_age"))
import dbn_smoke as D  # noqa: E402

from src.assessment.workers.client import serve  # noqa: E402

_MODEL = []


def brain_age(mni_paths):
    if not _MODEL:
        _MODEL.append(D.load_model())
    return {"pred_age": [D.predict(_MODEL[0], D.to_dbn_grid(p))[0] for p in mni_paths]}


if __name__ == "__main__":
    serve({"brain_age": brain_age, "ping": lambda: "dbn"})
