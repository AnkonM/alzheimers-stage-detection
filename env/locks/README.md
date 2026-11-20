# Environment lockfiles

Frozen 2026-10-22 from the local venvs with `importlib.metadata`, because pip is
missing in most of them. One file per venv. Each is the exact set the frozen
fusion-v3 system and its parity checks ran with.

| File | venv | Role |
|---|---|---|
| `mri.txt` | `.venv-mri` | m1 preprocessing (ANTsPy), ALZ CNN, LayerCAM |
| `deepbrainnet.txt` | `.venv-deepbrainnet` | brain age (TF 2.18, run with `TF_USE_LEGACY_KERAS=1`) |
| `fusion.txt` | `.venv-fusion` | CatBoost specialists, MRI specialist (sklearn 1.8.0 pickles), SHAP |
| `kan.txt` | `.venv-kan` | KAN fusion (pykan 0.2.8, CPU torch) |
| `synthseg.txt` | `.venv-synthseg` | SynthSeg 7.4.1 (FreeSurfer build pins) |

Notes for rebuilding:
- `torch==2.11.0+cu128` (mri) needs `--extra-index-url https://download.pytorch.org/whl/cu128`, and `torch==2.14.1+cpu` (kan) needs `.../whl/cpu`.
- Do not merge these environments. The sklearn pickles of the MRI specialist were written with 1.8.0, and the kan venv has 1.9.1.
- The lists include pip/setuptools where present. They are a record, not a resolver output, so install with `--no-deps` if pip complains.
