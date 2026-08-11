#!/usr/bin/env bash
set -euo pipefail
python src/icmi_bias_analysis.py \
  --adult data/Adult_Income_Cleaned.csv \
  --compas data/COMPAS_Cleaned.csv \
  --salary data/Job_Salary_Cleaned.csv \
  --outdir outputs
