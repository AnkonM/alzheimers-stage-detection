#!/usr/bin/env bash
# Start the assessment app: checks the environments and artifacts, then serves the API and the built frontend
# on 127.0.0.1 only (the four stage workers start with the server). Usage: ./run_app.sh [--check]
set -uo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"
LYP="$(cd ../.. && pwd)"
PORT="${AD_PORT:-8000}"
fail() { echo "ERROR: $1" >&2; exit 1; }

# 1. the five virtual environments (kept separate on purpose; see env/locks/README.md)
for v in .venv-app .venv-mri .venv-deepbrainnet .venv-fusion .venv-kan .venv-synthseg; do
  [ -x "$LYP/$v/bin/python" ] || fail "missing environment $LYP/$v. Rebuild it from env/locks (see env/locks/README.md); do not merge the environments."
done
PY="$LYP/.venv-app/bin/python"

# 2. model artifacts: every registry entry must exist and match its sha256
OUT="$("$PY" -m src.assessment.registry verify 2>&1)" || fail "artifact check failed: ${OUT##*$'\n'}
  Missing or changed model files are not served silently. Restore them (weights are distributed privately) and retry."
echo "artifacts: $OUT"

# 3. tools the pipeline calls
[ -f "$LYP/third_party/fs_synthseg/bin/mri_synthseg" ] || fail "SynthSeg not found at $LYP/third_party/fs_synthseg (needed for uploaded scans)."
[ -f frontend/dist/index.html ] || {
  command -v npm >/dev/null || fail "frontend/dist is missing and npm is not installed. Build it elsewhere: cd frontend && npm install && npm run build"
  echo "building the frontend…"; (cd frontend && npm install --silent && npm run build) || fail "frontend build failed"
}
[ -d demo_cases ] && echo "demo cases: $(ls demo_cases | wc -l) precomputed" || echo "note: no demo_cases/ folder (run: $PY -m src.assessment.cli precompute)"

[ "${1:-}" = "--check" ] && { echo "checks passed"; exit 0; }
echo "serving on http://127.0.0.1:$PORT (local only; Ctrl-C to stop)"
AD_PORT="$PORT" exec "$PY" -m src.api.app
