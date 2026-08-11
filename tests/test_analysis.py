from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from icmi_bias_analysis import (
    normalize_gender,
    stage1_by_gender_binary,
    compute_group_metrics_with_holdout,
    summarize_abs_gaps,
)


def test_normalize_gender():
    s = pd.Series([0, 1, "M", "F", "male", "female", "man", "woman"])
    got = normalize_gender(s).tolist()
    assert got == ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"]


def test_stage1_binary():
    df = pd.DataFrame({"gender": [0, 0, 1, 1], "income": [1, 0, 1, 1]})
    out = stage1_by_gender_binary(df, "gender", "income").set_index("group")
    assert int(out.loc["Male", "n"]) == 2
    assert float(out.loc["Male", "rate"]) == 0.5
    assert float(out.loc["Female", "rate"]) == 1.0


def test_group_metrics_smoke():
    rng = np.random.default_rng(123)
    n = 500
    gender = rng.integers(0, 2, size=n)
    age = rng.normal(40, 10, size=n)
    edu = rng.normal(10, 2, size=n)
    p = 1 / (1 + np.exp(-(-1 + 0.03 * age + 0.05 * edu - 0.2 * gender)))
    y = rng.binomial(1, p)
    df = pd.DataFrame(
        {
            "gender": gender,
            "income": y,
            "age": age,
            "Education Level Numeric": edu,
        }
    )
    metrics, holdout = compute_group_metrics_with_holdout(
        df,
        ["age", "Education Level Numeric"],
        "income",
        "gender",
        random_state=42,
    )
    assert set(metrics["group"]) == {"Male", "Female"}
    assert set(["SR", "TPR", "FPR", "n"]).issubset(metrics.columns)
    summary = summarize_abs_gaps(metrics, "Adult")
    assert 0 <= summary["SR_gap"] <= 1
    assert len(holdout["y_true"]) == len(holdout["y_pred"]) == len(holdout["group"])
