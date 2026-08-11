# Data Requirements

The cleaned datasets are not redistributed in this package.

Place the following files in this directory:

```text
Adult_Income_Cleaned.csv
COMPAS_Cleaned.csv
Job_Salary_Cleaned.csv
```

## Required columns

### Adult_Income_Cleaned.csv

- `gender`
- `income`
- `age`
- `Education Level Numeric`

Source-notebook row count: **42,166**

The analysis expects `income` to be binary 0/1 and uses the notebook convention `gender: 0=Male, 1=Female` when numeric coding is present.

### COMPAS_Cleaned.csv

- `Gender`
- `Hiring_Fit_Level_Num`
- `Age`

Source-notebook row count: **48,016**

The analysis creates `fit_binary = 1(Hiring_Fit_Level_Num >= 0.5)`.

### Job_Salary_Cleaned.csv

- `Gender`
- `Salary`
- `Age`
- `Education Level Numeric`

Source-notebook row count: **6,684**

The analysis creates `high_salary = 1(Salary >= median(Salary))`.

## Important limitation

The uploaded research notebook began from these cleaned files. It did not include the complete raw-data cleaning procedure, so this package cannot reconstruct these exact CSVs from raw data without additional preprocessing materials.

Do not add or redistribute datasets here unless their licenses permit redistribution.
