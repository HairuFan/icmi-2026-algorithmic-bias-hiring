# Reproducibility Audit

## Source reviewed

Primary source: the uploaded `ICMI 5th.ipynb` research notebook.

The notebook contains multiple generations of the same analysis, including exploratory/test cells and a final enhanced pipeline. The public release uses the final enhanced logic as the authoritative analysis path while preserving the numerical definitions used in the earlier cells.

## Findings and release changes

### 1. Hard-coded Google Drive dependency — fixed

The notebook mounted Google Drive and used personal paths such as a `Colab Notebooks/Social Justice` directory.

**Release change:** all paths are explicit CLI arguments or project-relative notebook paths.

### 2. Hidden notebook state and duplicate definitions — fixed

Functions and variables were redefined across several cells, and later cells depended on objects created by earlier cells. Running selected cells out of order could change behavior.

**Release change:** all analysis logic now lives in one importable module. The cleaned notebook calls that module.

### 3. Missing raw-data preprocessing — documented limitation

The notebook reads three already-cleaned CSV files, but it does not contain the transformations from the original public/raw datasets to those cleaned files.

**Release change:** the package explicitly claims analysis-stage reproducibility only. `data/README.md` documents the required schemas and expected row counts.

### 4. Silent feature fallback — fixed

The research notebook built feature lists with expressions such as “use this column if it exists.” A missing feature could therefore change the model without stopping execution.

**Release change:** the public pipeline requires the exact analysis columns and fails with a clear error when they are missing.

### 5. Gender normalization could silently retain unexpected labels — fixed

The research notebook normalized common Male/Female encodings but returned unknown values unchanged.

**Release change:** known encodings are normalized, while unknown non-missing values raise `ValueError`. The notebook's original convention `0=Male, 1=Female` is preserved and documented.

### 6. Implicit machine-learning defaults — fixed

`LogisticRegression(max_iter=1000)` depended on scikit-learn defaults that could change between releases.

**Release change:** `solver="lbfgs"`, `C=1.0`, `max_iter=1000`, and `random_state` are explicit. The train/test split remains stratified by gender with `test_size=0.30` and `random_state=42`, matching the source notebook.

### 7. Randomness — strengthened

The main split had a fixed seed and the final notebook used explicit bootstrap seeds.

**Release change:** all seeds are centralized, documented, and written to `run_metadata.json`.

### 8. Bootstrap p-value interpretation — clarified

The final notebook computes `2 * min(P(diff >= 0), P(diff <= 0))` from the bootstrap distribution.

**Release change:** the calculation is preserved, but documentation labels it as a sign-based approximate bootstrap diagnostic rather than a generic formal p-value.

### 9. Absolute-gap heatmap visual encoding — preserved, with optional alternative

The final notebook plotted non-negative absolute gaps with a diverging `RdBu_r` palette centered at zero and a range of -0.05 to 0.05. That visual choice is unusual for absolute magnitudes, but changing it would prevent exact figure-style reproduction.

**Release change:** the canonical `Figure2_FairnessGaps_Heatmap` preserves the notebook encoding. The pipeline additionally exports an explicitly labeled optional sequential version; numeric values are unchanged.

### 10. Regression CSV formatting — stabilized

The notebook exported `statsmodels.summary2().tables[1]`, whose formatting may vary across versions.

**Release change:** coefficient, standard error, statistic, p-value, and confidence limits are exported directly from fitted-model attributes.

### 11. Environment provenance — added

The source notebook did not record package versions.

**Release change:** `requirements.txt` pins the environment used to test this release, and each run writes installed versions, platform information, parameters, and SHA-256 input hashes to `run_metadata.json`.

## What was intentionally NOT changed

- Adult outcome definition (`income`).
- COMPAS-derived binary label: `Hiring_Fit_Level_Num >= 0.5`.
- Salary-derived binary label: `Salary >= dataset median`.
- Adult model features: `age`, `Education Level Numeric`.
- COMPAS model feature: `Age`.
- Salary model features: `Age`, `Education Level Numeric`.
- 70/30 split.
- gender-stratified split.
- fairness metrics SR / TPR / FPR.
- absolute gap definition as max minus min.
- `0.05` threshold used to label a gap "Substantive".
- baseline regression formulas.
- bootstrap resampling by group and seeds.

## Validation status

The package was syntax-checked and smoke-tested on synthetic data with the required schemas. The original cleaned CSV files were not part of the uploaded notebook, so exact numerical re-execution against the publication data cannot be performed in this environment.

The original notebook's stored outputs are preserved separately as validation targets in `EXPECTED_RESULTS.md`.
