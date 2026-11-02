# Lab 4: Data pipeline and quality

## 1. Problem definition

The task is to predict whether a person earns more than 50K dollars per year, using data from the 1994 US Census (UCI Adult dataset). This is a binary classification problem.

**Inputs.** 14 features about a person: 6 numeric (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) and 8 categorical (workclass, education, marital-status, occupation, relationship, race, sex, native-country).

**Output.** `income`, either `<=50K` or `>50K`. I treat `>50K` as the positive class.

**Success metrics.** Accuracy and F1. Only about a quarter of people earn more than 50K, so accuracy alone can look good even if the model mostly predicts `<=50K`. F1 shows how well the model finds the smaller `>50K` class.

**Constraints.**
- No data leakage: the train/test split is done before any cleaning, and every step that learns something from data (imputer, percentile clipping, scaling, one-hot categories) is fitted on train only and then applied to test.
- Reproducible: fixed seed 42, raw data is never modified, all settings are stored in `config.yaml`, the processed dataset has a SHA256 hash, and the whole pipeline runs with one command.
- Raw data is never silently changed: validation only reports problems, it does not drop rows.

## 2. Method

**One command:** `python lab04/src/run_lab04.py` (from repo root). Steps:

1. **Ingestion.** Downloads Adult with `fetch_openml(name='adult', version=2, as_frame=True)` once, saves an untouched copy to `data/raw/adult_raw.csv`, and always reads from that copy. Logs row and column count and the SHA256 of the raw file.
2. **Validation.** Checks the schema (int vs category types), ranges (age 17–90, hours 1–99, capital-gain 0–99999, capital-loss 0–4356), missing values (`?` or NaN), exact duplicates and label values. Writes `results/validation_report.md` and `.csv`. No rows are dropped.
3. **Split.** 70/30 stratified by income, seed 42, before cleaning.
4. **Cleaning** (one sklearn `Pipeline`, fitted on train only): `?` → missing, median imputing for numeric, mode for categorical, clipping numeric columns to the 1st/99th percentile of train, standard scaling, one-hot encoding with `handle_unknown='ignore'`. Exact duplicates are dropped from train only. Saved to `results/preprocessor.joblib`.
5. **Derived features.** `capital_net = capital-gain − capital-loss` and `hours_per_age = hours-per-week / (age − 16)`. Both are computed row by row, so they use no information from other rows and cannot leak. LogisticRegression is trained with and without them.
6. **Label quality.** Duplicate-conflict audit (same 14 features, different income) on the full data. 50 random rows checked against the data dictionary: ranges, both capital-gain and capital-loss > 0, Husband + Female, Wife + Male, education vs education-num mismatch, missing values. Saved to `results/label_audit_sample.csv`.
7. **Hash and docs.** SHA256 of the processed train and test arrays, saved with all settings to `results/config.yaml`. `data_card.md` is generated with source, license, size, biases and privacy notes.

## 3. Results

### Ingestion
- Rows: 48 842, columns: 15. Matches the expected size. Raw file SHA256: `d273d956c28eba72241f69dcbb78e37e93862248503a1873fb44b9fc2a6ae00b`.

### Validation (`results/validation_report.md`)

| Check | Column | Violations |
|---|---|---|
| missing | workclass | 2 799 |
| missing | occupation | 2 809 |
| missing | native-country | 857 |
| exact duplicate rows | all | 52 |
| all type, range and label checks | — | 0 |

### Cleaning
- Train 34 189 rows, test 14 653 rows. 24 exact duplicates dropped from train (34 165 left), test untouched.

### Derived features (`results/metrics.csv`)

| Features | Accuracy | F1 |
|---|---|---|
| 14 original | 0.8540 | 0.6601 |
| 14 + capital_net + hours_per_age | **0.8578** | **0.6696** |

### Label quality
- Duplicate-conflict audit: 5 groups of identical feature rows with different income labels (10 rows).
- 50-row manual check (`results/label_audit_sample.csv`): no impossible values. Ages and hours are in range, nobody has both capital-gain and capital-loss, Husband is always Male and Wife always Female, education always matches education-num. 3 of 50 rows have missing values; in two of them workclass and occupation are missing together.

### Reproducibility
- Processed dataset SHA256: `20c3eba1d3a53bfa06ab2f1d34edc0a1ab66ee155416dd73d2d9c8f40712f348` (stored in `results/config.yaml`).
- Running the pipeline twice gives the same hash.

## 4. Conclusions

1. The data is mostly clean. All type, range and label checks passed, so there are no impossible values like age 200 or negative hours. The real problems are missing values and duplicates: workclass (2 799), occupation (2 809) and native-country (857) have missing values, and there are 52 exact duplicate rows. Workclass and occupation are usually missing together, which makes sense, because if we don't know where a person works we also don't know their job. I only reported these problems in the validation step and did not drop anything there, so it is clear what was in the raw data.

2. The pipeline has no leakage. I split the data first and fitted the imputer, clipping and scaler only on train. Duplicates were removed only from train (24 rows), the test set stayed as it was. This way the test score shows how the model would work on new data it has never seen in any form.

3. The two derived features helped a little: accuracy went from 85.4% to 85.8% and F1 from 0.660 to 0.670. capital_net puts gain and loss into one number, and hours_per_age shows how much someone works compared to their age. The gain is small because logistic regression already sees the original columns. F1 is much lower than accuracy because only about 24% of people earn more than 50K, so the model has a harder time with that class.

4. The labels are not perfect. 5 groups of people (10 rows) have exactly the same 14 features but different income. No model can predict both correctly, so this is a limit on the best possible accuracy. It also shows that 14 features do not fully describe a person's income.

5. The whole pipeline runs with one command and gives the same SHA256 hash every time, because the seed, the settings and the raw data are fixed and saved.
