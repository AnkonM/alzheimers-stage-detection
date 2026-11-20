#!/usr/bin/env bash
# B2 / Experiment 2 fold-wise re-run (prereg B2). Order as the original EXP2 phases 1-4.
set -euo pipefail
cd "$(dirname "$0")/../../.."
LYP="$(cd ../.. && pwd)"
F="$LYP/.venv-fusion/bin/python"; K="$LYP/.venv-kan/bin/python"
export PYTHONPATH="$LYP/.pylib-exp2"
D=analysis/reviewer_response/exp2_foldwise/code
t0=$(date +%s)
$F $D/extract.py
$F $D/phase1_label_audit.py
$F $D/streams.py --tier MRI
for t in T0 T1 T2 T3 Full-spec S-T2+APOE4 Full-paper; do $F $D/streams.py --tier "$t"; done
for t in T0 T1 T2 T3 Full-spec S-T2+APOE4 Full-paper; do $K $D/kan_run.py --tier "$t"; done
$F $D/gate.py || true          # prereg B2: the gate is reported, not a stop condition
$F $D/evaluate.py
echo "EXP2 fold-wise total runtime $(( $(date +%s) - t0 )) s"
