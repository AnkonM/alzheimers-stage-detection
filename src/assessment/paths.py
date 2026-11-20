"""Repo and data-root locations. Registry paths are written relative to one of these roots."""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
# Large third-party files (DeepBrainNet weights) live beside the repo checkout, not in it.
LYP = pathlib.Path(os.environ.get("AD_LYP_ROOT", REPO.parent.parent))
ROOTS = {"repo": REPO, "lyp": LYP}
FROZEN = REPO / "analysis" / "fusion_v3" / "frozen"


def resolve(root: str, rel: str) -> pathlib.Path:
    return ROOTS[root] / rel
