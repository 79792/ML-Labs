# Validation report

Rows checked: 48842. No rows were dropped at this step.

| Check | Column | Violations | Status |
|---|---|---|---|
| type int | age | 0 | OK |
| type int | fnlwgt | 0 | OK |
| type int | education-num | 0 | OK |
| type int | capital-gain | 0 | OK |
| type int | capital-loss | 0 | OK |
| type int | hours-per-week | 0 | OK |
| type category | workclass | 0 | OK |
| type category | education | 0 | OK |
| type category | marital-status | 0 | OK |
| type category | occupation | 0 | OK |
| type category | relationship | 0 | OK |
| type category | race | 0 | OK |
| type category | sex | 0 | OK |
| type category | native-country | 0 | OK |
| type category | income | 0 | OK |
| range 17-90 | age | 0 | OK |
| range 1-99 | hours-per-week | 0 | OK |
| range 0-99999 | capital-gain | 0 | OK |
| range 0-4356 | capital-loss | 0 | OK |
| missing ('?' or NaN) | age | 0 | OK |
| missing ('?' or NaN) | workclass | 2799 | VIOLATION |
| missing ('?' or NaN) | fnlwgt | 0 | OK |
| missing ('?' or NaN) | education | 0 | OK |
| missing ('?' or NaN) | education-num | 0 | OK |
| missing ('?' or NaN) | marital-status | 0 | OK |
| missing ('?' or NaN) | occupation | 2809 | VIOLATION |
| missing ('?' or NaN) | relationship | 0 | OK |
| missing ('?' or NaN) | race | 0 | OK |
| missing ('?' or NaN) | sex | 0 | OK |
| missing ('?' or NaN) | capital-gain | 0 | OK |
| missing ('?' or NaN) | capital-loss | 0 | OK |
| missing ('?' or NaN) | hours-per-week | 0 | OK |
| missing ('?' or NaN) | native-country | 857 | VIOLATION |
| missing ('?' or NaN) | income | 0 | OK |
| exact duplicate rows | all | 52 | VIOLATION |
| label in {<=50K, >50K} | income | 0 | OK |
