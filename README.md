# ICMI 2026 Hiring-Bias Analysis — Reproducibility Package

This repository contains the cleaned, public-facing analysis code associated with:

> H. Fan and S. Wang, **“Diagnosing Algorithmic Bias in AI-Powered Hiring: Toward a Fairness-Aware Framework,”** 2026 IEEE 5th International Conference on Computing and Machine Intelligence (ICMI), Al-Ahsa, Saudi Arabia, 2026, pp. 1–5. DOI: **10.1109/ICMI68585.2026.11539889**


## Repository

Canonical repository URL: <https://github.com/HairuFan/icmi-2026-algorithmic-bias-hiring>

The GitHub repository and the Zenodo software record are intended to represent the same versioned research-software artifact. The IEEE DOI remains the citation for the conference paper itself.

## Reproducibility scope

This package reproduces the **analysis stage** found in the original research notebook. It starts from the three cleaned CSV files used by that notebook:

- `Adult_Income_Cleaned.csv`
- `COMPAS_Cleaned.csv`
- `Job_Salary_Cleaned.csv`

The original notebook did **not** contain the raw-data cleaning scripts. Therefore this release does **not** claim raw-to-results end-to-end reproducibility. See `data/README.md` and `REPRODUCIBILITY_AUDIT.md`.

The cleaned datasets are **not bundled** in this ZIP. Place your authorized copies in `data/`.

## What the analysis reproduces

1. Gender-group data-level summaries:
   - Adult: income > 50K rate
   - COMPAS-derived hiring-fit score: mean
   - Job Salary: mean salary
2. Logistic-regression holdout predictions and group fairness metrics:
   - Selection Rate (SR)
   - True Positive Rate (TPR)
   - False Positive Rate (FPR)
3. Absolute and signed Male–Female fairness gaps.
4. Group-stratified bootstrap confidence intervals and the notebook's sign-based approximate bootstrap p-value.
5. Baseline regression models:
   - Adult: Logit
   - COMPAS: Logit
   - Salary: OLS with HC1 robust standard errors
6. Publication-oriented figures and machine-readable CSV outputs.
7. `run_metadata.json` containing package versions, parameters, row counts, and SHA-256 hashes of input files.

## Quick start

### 1. Create an environment

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add the cleaned data

Place these files in `data/`:

```text
data/
├── Adult_Income_Cleaned.csv
├── COMPAS_Cleaned.csv
└── Job_Salary_Cleaned.csv
```

See `data/README.md` for required columns and source-notebook row-count checks.

### 3. Run the analysis

```bash
python src/icmi_bias_analysis.py \
  --adult data/Adult_Income_Cleaned.csv \
  --compas data/COMPAS_Cleaned.csv \
  --salary data/Job_Salary_Cleaned.csv \
  --outdir outputs
```

The default settings reproduce the source notebook's analysis choices:

- test fraction: `0.30`
- split random state: `42`
- split stratification: gender
- logistic regression: `solver=lbfgs`, `C=1.0`, `max_iter=1000`
- COMPAS binary threshold: hiring-fit score `>= 0.5`
- Salary binary threshold: dataset median salary
- fairness gap classification threshold: `0.05`
- bootstrap replicates: `2000`
- bootstrap seeds: Adult `42`, COMPAS `43`, Salary `44`
- gender coding convention inherited from the notebook: `0=Male`, `1=Female`

## Notebook

A cleaned notebook is provided at:

`notebooks/ICMI_5th_reproducible.ipynb`

It has no Google Drive dependency, no stored outputs, and no reliance on execution history. It calls the same functions as the command-line analysis, so there is a single source of truth.

## Expected validation targets

`EXPECTED_RESULTS.md` records the numerical values stored in the original notebook output. If you use the same cleaned CSV files, group counts and rounded fairness metrics should agree with those targets. Exact floating-point output can vary slightly across numerical-library versions.

## Important changes from the research notebook

The public release intentionally makes a small number of reproducibility-oriented changes without changing the core analytical definitions:

- hard-coded personal Google Drive paths removed;
- duplicate and exploratory/test cells removed;
- model defaults made explicit;
- required columns validated instead of silently dropped;
- unknown gender labels raise an error instead of being silently propagated;
- all random seeds are explicit;
- run parameters and input hashes are recorded;
- regression coefficient export no longer depends on `statsmodels.summary2()` formatting;
- the notebook's original absolute-gap heatmap style is preserved for strict reproduction; an additional file ending in `_Sequential_Optional` provides a more natural non-negative scale without changing numeric values.

For a line-by-line audit rationale, see `REPRODUCIBILITY_AUDIT.md`.

## Archival release

A DOI has been reserved for the version 1.0.0 archival release on Zenodo:

**Reserved Zenodo DOI:** `10.5281/zenodo.21894169`

The DOI will be registered when the Zenodo record is published.

## License

Code in this package is released under the MIT License. Dataset licenses remain with their original providers and are not modified by this repository.

## Citation

Please cite the IEEE paper when using the scientific results. The software package also contains `CITATION.cff` for code citation.

**Paper DOI:** `10.1109/ICMI68585.2026.11539889`
