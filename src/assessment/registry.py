"""Artifact registry: logical name -> file + sha256. A missing or changed file is a hard error.

`python -m src.assessment.registry build` regenerates artifacts/registry.json from the freeze manifest
(frozen/MANIFEST.sha256, written at freeze 8f351b6) plus the CNN checkpoints, DeepBrainNet weights and atlas.
`python -m src.assessment.registry verify` checks every entry.
"""
import hashlib
import json
import pathlib
import sys

from .paths import FROZEN, REPO, resolve

REGISTRY = pathlib.Path(__file__).parent / "artifacts" / "registry.json"
EXTRA = [  # (name, root, relative path)
    ("dbn_weights", "lyp", "third_party/DeepBrainNet/Models/DBN_model.h5"),
    ("atlas96", "repo", "analysis/explainability/atlas/atlas96.nii.gz"),
    ("atlas_regions", "repo", "analysis/explainability/atlas/regions.json"),
] + [(f"alz_fold{f}_s{s}", "repo", f"analysis/fusion_feasibility/runs/fold{f}_s{s}/best.pt")
     for f in range(5) for s in (42, 43, 44)]


class ArtifactError(RuntimeError):
    pass


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> dict:
    entries = []
    for line in (FROZEN / "MANIFEST.sha256").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        entries.append({"name": pathlib.Path(rel).name, "root": "repo", "path": rel.strip(), "sha256": digest})
    for name, root, rel in EXTRA:
        entries.append({"name": name, "root": root, "path": rel, "sha256": sha256(resolve(root, rel))})
    return {"freeze_commit": "8f351b6", "artifacts": entries}


def load() -> dict:
    return json.loads(REGISTRY.read_text())


def verify(names: set[str] | None = None) -> list[dict]:
    """Return [{name, sha256}] for the checked artifacts; raise ArtifactError naming any problem."""
    out, problems = [], []
    for e in load()["artifacts"]:
        if names is not None and e["name"] not in names:
            continue
        p = resolve(e["root"], e["path"])
        if not p.exists():
            problems.append(f"missing: {e['name']} ({e['root']}:{e['path']})")
        elif sha256(p) != e["sha256"]:
            problems.append(f"sha256 mismatch: {e['name']}")
        else:
            out.append({"name": e["name"], "sha256": e["sha256"]})
    if problems:
        raise ArtifactError("; ".join(problems))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "build":
        REGISTRY.write_text(json.dumps(build(), indent=1) + "\n")
        print("wrote", REGISTRY.relative_to(REPO))
    else:
        print(len(verify()), "artifacts verified")
