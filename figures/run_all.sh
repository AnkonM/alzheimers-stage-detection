#!/usr/bin/env bash
# Build every paper figure into figures/out/ (never committed). Stops on the first failed check.
# Matplotlib figures: LYP/.venv-mri; Figure 5 (playwright + pydantic): LYP/.venv-app.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LYP="$(cd "$HERE/../../.." && pwd)"
export MAIN_CHECKOUT="${MAIN_CHECKOUT:-$(cd "$HERE/../../ad-detection" && pwd)}"
cd "$HERE/scripts"
for s in fig1_mri_input fig2_test_confusion fig3_cascade fig4_mri_shap S1_brainage S2_faithfulness S3_exp2_confusion; do
  echo "== $s"; "$LYP/.venv-mri/bin/python" "$s.py"
done
echo "== fig5_report_card"; "$LYP/.venv-app/bin/python" fig5_report_card.py
