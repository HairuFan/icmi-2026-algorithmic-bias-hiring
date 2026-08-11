#!/usr/bin/env python3
"""Reproduce the analysis stage for Fan & Wang (ICMI 2026).

Associated paper DOI: 10.1109/ICMI68585.2026.11539889
Software DOI: 10.5281/zenodo.21894169

The pipeline starts from the three cleaned CSV files used in the source
notebook. Raw-to-clean preprocessing was not present in that notebook and is
outside this package's reproducibility claim.
"""
from __future__ import annotations

import argparse, hashlib, json, platform, sys, warnings
from importlib import metadata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

VERSION = "1.0.0"
PAPER_DOI = "10.1109/ICMI68585.2026.11539889"
SOFTWARE_DOI = "10.5281/zenodo.21894169"
RANDOM_STATE, TEST_SIZE, BOOTSTRAP_B = 42, 0.30, 2000
EXPECTED_ROWS = {"Adult": 42166, "COMPAS": 48016, "Salary": 6684}
REQUIRED_COLUMNS = {
    "Adult": ["gender", "income", "age", "Education Level Numeric"],
    "COMPAS": ["Gender", "Hiring_Fit_Level_Num", "Age"],
    "Salary": ["Gender", "Salary", "Age", "Education Level Numeric"],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions(names):
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def normalize_gender(series: pd.Series) -> pd.Series:
    """Normalize supported gender encodings to exactly Male/Female."""
    male, female = {"0", "m", "male", "man"}, {"1", "f", "female", "woman"}
    def one(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            s = str(int(v))
        elif isinstance(v, (float, np.floating)) and float(v).is_integer():
            s = str(int(v))
        else:
            s = str(v).strip().lower()
        if s in male: return "Male"
        if s in female: return "Female"
        raise ValueError(f"Unrecognized gender value {v!r}; expected 0/1, M/F, male/female, or man/woman.")
    return series.map(one)


def validate_columns(df, required, dataset):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset}: missing required columns {missing}")


def validate_binary(series, dataset="dataset", column="outcome"):
    vals = set(pd.to_numeric(pd.Series(series).dropna(), errors="raise").unique())
    if not vals or not vals.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"{dataset}.{column} must be binary 0/1; observed {sorted(vals)}")


def stage1_by_gender_binary(df, gender_col, outcome_col):
    d = df[[gender_col, outcome_col]].dropna().copy()
    d[gender_col] = normalize_gender(d[gender_col])
    validate_binary(d[outcome_col], column=outcome_col)
    d[outcome_col] = d[outcome_col].astype(int)
    return (d.groupby(gender_col, sort=True)[outcome_col].agg(n="count", rate="mean")
             .reset_index().rename(columns={gender_col: "group"}))


def stage1_by_gender_numeric(df, gender_col, outcome_col):
    d = df[[gender_col, outcome_col]].dropna().copy()
    d[gender_col] = normalize_gender(d[gender_col])
    d[outcome_col] = pd.to_numeric(d[outcome_col], errors="raise")
    return (d.groupby(gender_col, sort=True)[outcome_col].agg(n="count", mean="mean")
             .reset_index().rename(columns={gender_col: "group"}))


def _metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sr = float(np.mean(y_pred))
    tpr = float(tp / (tp + fn)) if tp + fn else np.nan
    fpr = float(fp / (fp + tn)) if fp + tn else np.nan
    return sr, tpr, fpr


def compute_group_metrics_with_holdout(df, features, outcome_col, gender_col,
                                       *, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    features = list(features)
    required = features + [outcome_col, gender_col]
    validate_columns(df, required, "model input")
    d = df[required].dropna().copy()
    d[gender_col] = normalize_gender(d[gender_col])
    if d[gender_col].nunique() != 2:
        raise ValueError("Both Male and Female groups are required.")
    X = d[features].apply(pd.to_numeric, errors="raise").to_numpy(float)
    y = d[outcome_col].astype(int).to_numpy()
    validate_binary(y, column=outcome_col)
    if np.unique(y).size != 2:
        raise ValueError("Model outcome must contain both classes.")
    g = d[gender_col].to_numpy()
    Xtr, Xte, ytr, yte, _, gte = train_test_split(
        X, y, g, test_size=test_size, random_state=random_state, stratify=g)
    clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, random_state=random_state)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    rows = []
    for group in ["Male", "Female"]:
        m = gte == group
        if not m.any(): raise ValueError(f"No {group} cases in holdout set.")
        sr, tpr, fpr = _metrics(yte[m], pred[m])
        rows.append({"group": group, "SR": sr, "TPR": tpr, "FPR": fpr, "n": int(m.sum())})
    return pd.DataFrame(rows), {"y_true": yte, "y_pred": pred, "group": gte}


def summarize_abs_gaps(metrics, dataset_name, threshold=0.05):
    gaps = {m: float(metrics[m].max() - metrics[m].min()) for m in ["SR", "TPR", "FPR"]}
    return {"Dataset": dataset_name, "SR_gap": gaps["SR"], "TPR_gap": gaps["TPR"],
            "FPR_gap": gaps["FPR"], "Key Bias": "Substantive" if max(gaps.values()) >= threshold else "Minor"}


def summarize_signed_gaps(metrics, dataset_name):
    def v(group, metric):
        x = metrics.loc[metrics.group == group, metric]
        if len(x) != 1: raise ValueError(f"Expected one {group} row for {metric}")
        return float(x.iloc[0])
    out = {"Dataset": dataset_name}
    for metric in ["SR", "TPR", "FPR"]:
        out[f"{metric}_diff"] = v("Male", metric) - v("Female", metric)
    return out


def bootstrap_gap_ci(holdout, *, B=BOOTSTRAP_B, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y, pred, g = holdout["y_true"], holdout["y_pred"], holdout["group"]
    im, iff = np.where(g == "Male")[0], np.where(g == "Female")[0]
    diffs = {k: np.empty(B) for k in ["SR", "TPR", "FPR"]}
    for b in range(B):
        m = rng.choice(im, len(im), replace=True); f = rng.choice(iff, len(iff), replace=True)
        mm, ff = _metrics(y[m], pred[m]), _metrics(y[f], pred[f])
        for j, k in enumerate(["SR", "TPR", "FPR"]): diffs[k][b] = mm[j] - ff[j]
    out = {}
    for k, a in diffs.items():
        finite = a[np.isfinite(a)]
        out[f"{k}_ci_low"], out[f"{k}_ci_high"] = np.quantile(finite, [0.025, 0.975])
        out[f"{k}_bootstrap_p"] = min(1.0, 2 * min(np.mean(finite >= 0), np.mean(finite <= 0)))
    return {k: float(v) for k, v in out.items()}


def make_combined_plot(adult, compas, salary, outpath):
    datasets = [("Adult", adult), ("COMPAS", compas), ("Salary", salary)]
    metrics = ["SR", "TPR", "FPR"]; width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.1), sharey=True)
    for ax, (title, d) in zip(axes, datasets):
        x = np.arange(3)
        for off, group in [(-width/2, "Male"), (width/2, "Female")]:
            vals = [float(d.loc[d.group == group, m].iloc[0]) for m in metrics]
            ax.bar(x + off, vals, width, label=group)
        ax.set_title(title); ax.set_xticks(x, metrics); ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .88))
    outpath = Path(outpath); fig.savefig(outpath, dpi=600, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def make_abs_gap_heatmap(table2, outpath, preserve_notebook_style=True):
    x = table2.set_index("Dataset")[["SR_gap", "TPR_gap", "FPR_gap"]]
    fig, ax = plt.subplots(figsize=(5.5, 2.7))
    if preserve_notebook_style:
        sns.heatmap(x, annot=True, fmt=".3f", cmap="RdBu_r", center=0, vmin=-.05, vmax=.05, ax=ax)
    else:
        sns.heatmap(x, annot=True, fmt=".3f", cmap="Blues", vmin=0, ax=ax)
    ax.set_xlabel("Fairness metric gap"); ax.set_ylabel(""); fig.tight_layout()
    outpath = Path(outpath); fig.savefig(outpath, dpi=600, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def save_coef_table(result, path):
    ci = result.conf_int()
    pd.DataFrame({"term": result.params.index, "coef": result.params.values,
                  "std_err": result.bse.values, "statistic": result.tvalues.values,
                  "p_value": result.pvalues.values, "ci_low": ci.iloc[:,0].values,
                  "ci_high": ci.iloc[:,1].values}).to_csv(path, index=False)


def run_analysis(adult_path, compas_path, salary_path, outdir,
                 *, test_size=TEST_SIZE, random_state=RANDOM_STATE, bootstrap_B=BOOTSTRAP_B):
    adult_path, compas_path, salary_path, outdir = map(Path, [adult_path, compas_path, salary_path, outdir])
    outdir.mkdir(parents=True, exist_ok=True)
    for p in [adult_path, compas_path, salary_path]:
        if not p.is_file(): raise FileNotFoundError(p)
    adult, compas, salary = map(pd.read_csv, [adult_path, compas_path, salary_path])
    for name, df in [("Adult", adult), ("COMPAS", compas), ("Salary", salary)]:
        validate_columns(df, REQUIRED_COLUMNS[name], name)
        if len(df) != EXPECTED_ROWS[name]:
            warnings.warn(f"{name}: {len(df)} rows; source notebook has {EXPECTED_ROWS[name]}. Results may differ.", RuntimeWarning)

    a1 = stage1_by_gender_binary(adult, "gender", "income")
    c1 = stage1_by_gender_numeric(compas, "Gender", "Hiring_Fit_Level_Num")
    s1 = stage1_by_gender_numeric(salary, "Gender", "Salary")
    a1.to_csv(outdir/"table1_adult.csv", index=False); c1.to_csv(outdir/"table1_compas.csv", index=False); s1.to_csv(outdir/"table1_salary.csv", index=False)

    cb = compas.copy(); cb["fit_binary"] = (pd.to_numeric(cb.Hiring_Fit_Level_Num) >= .5).astype(int)
    sb = salary.copy(); salary_median = float(pd.to_numeric(sb.Salary).median()); sb["high_salary"] = (pd.to_numeric(sb.Salary) >= salary_median).astype(int)
    am, ah = compute_group_metrics_with_holdout(adult, ["age","Education Level Numeric"], "income", "gender", test_size=test_size, random_state=random_state)
    cm, ch = compute_group_metrics_with_holdout(cb, ["Age"], "fit_binary", "Gender", test_size=test_size, random_state=random_state)
    smet, sh = compute_group_metrics_with_holdout(sb, ["Age","Education Level Numeric"], "high_salary", "Gender", test_size=test_size, random_state=random_state)
    for name, d in [("adult",am),("compas",cm),("salary",smet)]: d.to_csv(outdir/f"group_metrics_{name}.csv", index=False)

    t2 = pd.DataFrame([summarize_abs_gaps(am,"Adult"), summarize_abs_gaps(cm,"COMPAS"), summarize_abs_gaps(smet,"Salary")])
    signed = pd.DataFrame([summarize_signed_gaps(am,"Adult"), summarize_signed_gaps(cm,"COMPAS"), summarize_signed_gaps(smet,"Salary")])
    t2.to_csv(outdir/"table2_bias_summary_abs.csv", index=False); signed.to_csv(outdir/"table2_gender_diffs_signed.csv", index=False)
    cis = pd.DataFrame([{"Dataset":"Adult", **bootstrap_gap_ci(ah,B=bootstrap_B,seed=42)},
                        {"Dataset":"COMPAS", **bootstrap_gap_ci(ch,B=bootstrap_B,seed=43)},
                        {"Dataset":"Salary", **bootstrap_gap_ci(sh,B=bootstrap_B,seed=44)}])
    t3 = signed.merge(cis, on="Dataset"); t3.to_csv(outdir/"table3_gender_diffs_CI_p.csv", index=False)
    make_combined_plot(am,cm,smet,outdir/"Figure1_Combined_ModelOutcomeDisparities.png")
    make_abs_gap_heatmap(t2,outdir/"Figure2_FairnessGaps_Heatmap.png",True)
    make_abs_gap_heatmap(t2,outdir/"Figure2_FairnessGaps_Heatmap_Sequential_Optional.png",False)

    def gd(df, col):
        d=df.copy(); d[col]=normalize_gender(d[col]); d["female"]=(d[col]=="Female").astype(int); return d
    ar=gd(adult,"gender").dropna(subset=["income","female","age","Education Level Numeric"])
    lar=sm.Logit(ar.income.astype(int), sm.add_constant(ar[["female","age","Education Level Numeric"]])).fit(disp=False); save_coef_table(lar,outdir/"reg_logit_adult_coef.csv")
    cr=gd(cb,"Gender").dropna(subset=["fit_binary","female","Age"])
    lcr=sm.Logit(cr.fit_binary.astype(int), sm.add_constant(cr[["female","Age"]])).fit(disp=False); save_coef_table(lcr,outdir/"reg_logit_compas_coef.csv")
    sr=gd(salary,"Gender").dropna(subset=["Salary","female","Age","Education Level Numeric"])
    osr=sm.OLS(sr.Salary.astype(float), sm.add_constant(sr[["female","Age","Education Level Numeric"]])).fit(cov_type="HC1"); save_coef_table(osr,outdir/"reg_ols_salary_coef.csv")
    reg = pd.DataFrame([{"Dataset":"Adult","Model":"Logit","female_coef":float(lar.params.female),"female_p":float(lar.pvalues.female)},
                        {"Dataset":"COMPAS","Model":"Logit","female_coef":float(lcr.params.female),"female_p":float(lcr.pvalues.female)},
                        {"Dataset":"Salary","Model":"OLS_HC1","female_coef":float(osr.params.female),"female_p":float(osr.pvalues.female)}])
    reg.to_csv(outdir/"table4_baseline_reg_summary.csv", index=False)

    meta={"package_version":VERSION,"paper_doi":PAPER_DOI,"software_doi":SOFTWARE_DOI,"python":sys.version,"platform":platform.platform(),
          "packages":package_versions(["numpy","pandas","scikit-learn","matplotlib","seaborn","statsmodels"]),
          "parameters":{"test_size":test_size,"random_state":random_state,"bootstrap_B":bootstrap_B,"bootstrap_seeds":{"Adult":42,"COMPAS":43,"Salary":44},"compas_binary_threshold":.5,"salary_binary_threshold":salary_median,"fairness_gap_threshold":.05,"gender_encoding":"0=Male, 1=Female"},
          "inputs":{"Adult":{"path":str(adult_path),"rows":len(adult),"sha256":sha256_file(adult_path)},"COMPAS":{"path":str(compas_path),"rows":len(compas),"sha256":sha256_file(compas_path)},"Salary":{"path":str(salary_path),"rows":len(salary),"sha256":sha256_file(salary_path)}}}
    (outdir/"run_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return {"adult_table1":a1,"compas_table1":c1,"salary_table1":s1,"adult_metrics":am,"compas_metrics":cm,"salary_metrics":smet,"table2_abs":t2,"table2_signed":signed,"table3_ci":t3,"reg_summary":reg,"run_metadata":meta}


def build_parser():
    p=argparse.ArgumentParser(description="Reproduce the ICMI 2026 hiring-bias analysis from cleaned CSV files.")
    p.add_argument("--adult",type=Path,required=True); p.add_argument("--compas",type=Path,required=True); p.add_argument("--salary",type=Path,required=True)
    p.add_argument("--outdir",type=Path,default=Path("outputs")); p.add_argument("--test-size",type=float,default=TEST_SIZE); p.add_argument("--random-state",type=int,default=RANDOM_STATE); p.add_argument("--bootstrap",type=int,default=BOOTSTRAP_B)
    return p


def main():
    a=build_parser().parse_args(); run_analysis(a.adult,a.compas,a.salary,a.outdir,test_size=a.test_size,random_state=a.random_state,bootstrap_B=a.bootstrap); print(f"Analysis complete. Outputs written to: {a.outdir.resolve()}")

if __name__ == "__main__": main()
