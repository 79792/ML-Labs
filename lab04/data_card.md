# Data card: UCI Adult (Census Income)

## Source
- UCI Machine Learning Repository, "Adult" dataset (Becker & Kohavi, 1996), extracted from the 1994 US Census database.
- Loaded with `sklearn.datasets.fetch_openml(name='adult', version=2)` (OpenML ID 1590).
- License: Creative Commons Attribution 4.0 (CC BY 4.0).

## Size and structure
- 48842 rows, 15 columns: 14 features (6 numeric, 8 categorical) + label `income` (`<=50K` / `>50K`).
- Class balance: 23.9% `>50K`, 76.1% `<=50K` (imbalanced, so F1 is reported next to accuracy).
- Missing values (`?`/NaN): {'workclass': 2799, 'occupation': 2809, 'native-country': 857}.
- Exact duplicate rows: 52. Feature-duplicate groups with conflicting labels: 5 (10 rows).

## Processing
- 70/30 stratified split (seed 42) before any cleaning. Imputer (median / mode), 1st–99th percentile clipping and scaling are fitted on train only. Duplicates dropped from train only.
- Derived features: `capital_net`, `hours_per_age`.
- Raw file SHA256: `d273d956c28eba72241f69dcbb78e37e93862248503a1873fb44b9fc2a6ae00b`
- Processed dataset SHA256: `20c3eba1d3a53bfa06ab2f1d34edc0a1ab66ee155416dd73d2d9c8f40712f348`

## Known biases
- Data is from 1994; incomes are not adjusted for inflation, so the 50K threshold means something different today.
- Sex is imbalanced: {'Male': 0.668, 'Female': 0.332}.
- Race is very imbalanced: {'White': 0.855, 'Black': 0.096, 'Asian-Pac-Islander': 0.031, 'Amer-Indian-Eskimo': 0.01, 'Other': 0.008}. A model can be much less reliable for small groups.
- `native-country` is mostly United-States, other countries have very few rows.
- `fnlwgt` is a census sampling weight, not a personal attribute.
- The label is binary and self-reported; models trained on it can learn historical pay gaps (for example by sex or race).

## Privacy concerns
- No names or IDs, but it has sensitive attributes: sex, race, marital status, relationship, native country.
- Rare combinations (for example age + country + occupation) could make some people re-identifiable.
- Sensitive attributes should not be used for real decisions about people (credit, hiring) without a fairness review.

## Intended use
- Teaching and benchmarking tabular ML pipelines. Not for real income or credit decisions.
- Baseline (LogisticRegression): accuracy 0.8578, F1 0.6696 with derived features.
