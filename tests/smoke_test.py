#!/usr/bin/env python3
"""Generate synthetic datasets and run the full pipeline as a smoke test."""
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icmi_bias_analysis import run_analysis


def make_data(seed=7):
    rng = np.random.default_rng(seed)

    def gender(n):
        return rng.integers(0, 2, size=n)

    n = 800
    g = gender(n)
    age = rng.normal(40, 10, n)
    edu = rng.normal(10, 2, n)
    p = 1 / (1 + np.exp(-(-1.5 + 0.035 * age + 0.06 * edu - 0.25 * g)))
    adult = pd.DataFrame({
        "gender": g,
        "income": rng.binomial(1, p),
        "age": age,
        "Education Level Numeric": edu,
    })

    n = 900
    g = gender(n)
    age = rng.normal(35, 9, n)
    latent = 0.30 + 0.005 * age - 0.05 * g + rng.normal(0, 0.15, n)
    compas = pd.DataFrame({
        "Gender": g,
        "Hiring_Fit_Level_Num": np.clip(latent, 0, 1),
        "Age": age,
    })

    n = 700
    g = gender(n)
    age = rng.normal(38, 8, n)
    edu = rng.normal(11, 2, n)
    salary = 50000 + 1100 * age + 3500 * edu - 5000 * g + rng.normal(0, 12000, n)
    salary_df = pd.DataFrame({
        "Gender": g,
        "Salary": salary,
        "Age": age,
        "Education Level Numeric": edu,
    })
    return adult, compas, salary_df


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        adult, compas, salary = make_data()
        adult.to_csv(td / "adult.csv", index=False)
        compas.to_csv(td / "compas.csv", index=False)
        salary.to_csv(td / "salary.csv", index=False)
        out = td / "out"
        result = run_analysis(
            td / "adult.csv",
            td / "compas.csv",
            td / "salary.csv",
            out,
            bootstrap_B=100,
        )
        required = [
            "table1_adult.csv",
            "table1_compas.csv",
            "table1_salary.csv",
            "group_metrics_adult.csv",
            "group_metrics_compas.csv",
            "group_metrics_salary.csv",
            "table2_bias_summary_abs.csv",
            "table3_gender_diffs_CI_p.csv",
            "table4_baseline_reg_summary.csv",
            "Figure1_Combined_ModelOutcomeDisparities.png",
            "Figure2_FairnessGaps_Heatmap.png",
            "run_metadata.json",
        ]
        missing = [name for name in required if not (out / name).exists()]
        if missing:
            raise SystemExit(f"Smoke test failed. Missing: {missing}")
        print("Full-pipeline smoke test passed.")
