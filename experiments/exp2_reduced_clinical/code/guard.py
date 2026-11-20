"""Frozen-artefact guard (any env). Read-only on analysis/.

  python guard.py snapshot <label>   verify frozen/MANIFEST.sha256 and record sha256 of
                                     config_log.csv, kan_runs/*, streams.npz -> logs/hashes_<label>.json
  python guard.py compare <a> <b>    compare two snapshots
"""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(label):
    r = subprocess.run(["sha256sum", "-c", "--quiet", "analysis/fusion_v3/frozen/MANIFEST.sha256"], cwd=C.ROOT,
                       capture_output=True, text=True)
    n = sum(1 for _ in open(os.path.join(C.V3, "frozen", "MANIFEST.sha256")))
    files = ["config_log.csv", "streams.npz"] + [os.path.join("kan_runs", f) for f in sorted(os.listdir(os.path.join(C.V3, "kan_runs")))]
    snap = {"manifest_ok": r.returncode == 0, "manifest_files": n, "manifest_errors": r.stdout + r.stderr,
            "files": {f: sha(os.path.join(C.V3, f)) for f in files}}
    C.write_json(snap, os.path.join(C.LOGS, f"hashes_{label}.json"))
    print(f"{label}: MANIFEST {'OK' if snap['manifest_ok'] else 'FAILED'} ({n} files); {len(files)} paper outputs hashed")
    return snap


def compare(a, b):
    A, B = (json.load(open(os.path.join(C.LOGS, f"hashes_{x}.json"))) for x in (a, b))
    changed = sorted(k for k in set(A["files"]) | set(B["files"]) if A["files"].get(k) != B["files"].get(k))
    ok = A["manifest_ok"] and B["manifest_ok"] and not changed
    print(f"{a} vs {b}: {'UNCHANGED' if ok else 'CHANGED'}; changed files: {changed}")
    return ok, changed


if __name__ == "__main__":
    if sys.argv[1] == "snapshot":
        snapshot(sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])
