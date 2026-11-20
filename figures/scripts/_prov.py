"""Provenance helper shared by all figure scripts: input hashing, git commit, recorded assertions.

Each script builds one Prov, records every input (path + sha256) and every check (name, expected,
observed, pass), and writes figures/out/provenance/<figure>.json. A failed check raises after it is
recorded, so the run stops and the failure stays on disk.
"""
import hashlib
import json
import os
import subprocess
import sys

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # worktree root
MAIN = os.environ.get("MAIN_CHECKOUT", os.path.join(os.path.dirname(WT), "ad-detection"))  # read-only inputs
OUT = os.path.join(WT, "figures", "out")
sys.path.insert(0, os.path.join(WT, "figures"))


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git(*args, cwd=WT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True).stdout


class Prov:
    def __init__(self, figure, script):
        self.d = {"figure": figure, "script": os.path.relpath(script, WT), "inputs": [], "checks": [], "notes": []}
        rel = self.d["script"]
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel, "figures/paper_style.py",
                                "figures/scripts/_prov.py"], cwd=WT).returncode
        self.d["commit"] = git("rev-parse", "--short", "HEAD").decode().strip()
        self.d["branch"] = git("rev-parse", "--abbrev-ref", "HEAD").decode().strip()
        self.d["script_committed_unmodified"] = dirty == 0

    def file(self, path, label=None, base=WT):
        """Read a file by path (relative to `base`), record its sha256, return its bytes."""
        full = path if os.path.isabs(path) else os.path.join(base, path)
        b = open(full, "rb").read()
        self.d["inputs"].append({"path": label or os.path.relpath(full, WT if base == WT else os.path.dirname(base)),
                                 "sha256": sha256_bytes(b), "bytes": len(b)})
        return b

    def git_blob(self, ref, path):
        """Read a committed file from another ref (read-only), record ref:path, blob id and sha256."""
        b = git("show", f"{ref}:{path}")
        blob = git("rev-parse", f"{ref}:{path}").decode().strip()
        self.d["inputs"].append({"path": f"{ref}:{path}", "git_blob": blob, "sha256": sha256_bytes(b), "bytes": len(b)})
        return b

    def check(self, name, ok, expected=None, observed=None):
        self.d["checks"].append({"name": name, "pass": bool(ok), "expected": _j(expected), "observed": _j(observed)})
        if not ok:
            self.write()
            raise AssertionError(f"[{self.d['figure']}] check failed: {name}: expected {expected}, observed {observed}")

    def note(self, text):
        self.d["notes"].append(text)

    def write(self, outputs=()):
        self.d["outputs"] = [os.path.basename(o) for o in outputs]
        os.makedirs(os.path.join(OUT, "provenance"), exist_ok=True)
        with open(os.path.join(OUT, "provenance", f"{self.d['figure']}.json"), "w") as f:
            json.dump(self.d, f, indent=1)


def _j(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)
