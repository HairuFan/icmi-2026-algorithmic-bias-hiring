# Expected Results from the Original Notebook

These values were stored in the uploaded research notebook output. They provide a practical validation target when the same cleaned CSV files are supplied.

## Stage 1 — data-level summaries

### Adult income

| group | n | rate |
|---|---:|---:|
| Female | 14,057 | 0.111830 |
| Male | 28,109 | 0.312782 |

### COMPAS-derived hiring-fit score

| group | n | mean |
|---|---:|---:|
| Female | 10,708 | 0.384012 |
| Male | 37,308 | 0.507451 |

### Job Salary

| group | n | mean salary |
|---|---:|---:|
| Female | 3,013 | 108089.845005 |
| Male | 3,671 | 121069.266412 |

## Model outcome metrics

### Adult

| group | SR | TPR | FPR | holdout n |
|---|---:|---:|---:|---:|
| Male | 0.109332 | 0.234316 | 0.052186 | 8,433 |
| Female | 0.075172 | 0.236626 | 0.054141 | 4,217 |

### COMPAS

| group | SR | TPR | FPR | holdout n |
|---|---:|---:|---:|---:|
| Male | 0.044939 | 0.076640 | 0.026818 | 11,193 |
| Female | 0.028954 | 0.055495 | 0.018317 | 3,212 |

### Salary

| group | SR | TPR | FPR | holdout n |
|---|---:|---:|---:|---:|
| Male | 0.500907 | 0.782748 | 0.130252 | 1,102 |
| Female | 0.464602 | 0.805164 | 0.161088 | 904 |

## Absolute fairness gaps

| Dataset | SR gap | TPR gap | FPR gap | notebook label |
|---|---:|---:|---:|---|
| Adult | 0.034160 | 0.002310 | 0.001955 | Minor |
| COMPAS | 0.015985 | 0.021145 | 0.008502 | Minor |
| Salary | 0.036306 | 0.022417 | 0.030836 | Minor |

## Validation guidance

- Group counts should match exactly if the same cleaned CSV files are used.
- Metrics rounded to six decimals should normally match.
- Small floating-point differences can occur across numerical-library versions.
- If results differ materially, first compare `run_metadata.json`, input SHA-256 hashes, row counts, column names, and gender encoding.
