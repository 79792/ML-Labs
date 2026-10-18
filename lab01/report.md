# Lab 1: Environment and first system measurements

## 1. Goal

Set up a reproducible Python environment and measure the system cost (training time, inference latency, model size, peak memory) of two baseline classifiers on Breast Cancer Wisconsin, then decide which deployment tier (Cloud / Edge / Mobile / TinyML) each model fits.

## 2. Method

- Environment: isolated venv, pinned `requirements.txt`; versions saved in `results/versions.txt` by `src/print_versions.py`.
- Data: `load_breast_cancer`, 70/30 stratified split, `random_state=42`; seeds set for `random`, `numpy`, `torch`.
- Models: `LogisticRegression(max_iter=1000, random_state=42)` and `RandomForestClassifier(n_estimators=100, random_state=42)`.
- Training time: 1 warm-up, then median of 5 fits on fresh clones (`time.perf_counter`).
- Inference latency: one test sample, 1 warm-up, median of 100 `predict` calls.
- Model size: `joblib.dump` per model (`results/model_<name>.joblib`; both also in `results/model.joblib`).
- Peak memory: `memory_profiler.memory_usage(max_usage=True)` = peak process RSS during fit and predict; delta vs. baseline RSS also reported.
- Deployment fit: model is placed in a tier if inference peak RSS ≤ tier max memory, latency ≤ tier limit and size ≤ tier limit.

## 3. Results

**Accuracy** (`results/baseline_accuracy.csv`)

| Model | Test accuracy |
|---|---|
| LogisticRegression | 0.9474 |
| RandomForest | 0.9357 |

**System cost** (`results/system_cost.csv`)

| Model | Train (ms, median) | Inference (ms, median) | Size (bytes / KB) | Peak RSS train (MiB) | Peak RSS infer (MiB) |
|---|---|---|---|---|---|
| LogisticRegression | 407.314 | 0.0193 | 1 055 / 1.03 | 265.4 | 265.4 |
| RandomForest | 67.470 | 0.9121 | 290 889 / 284.07 | 266.7 | 266.7 |

**Deployment fit** (`results/deployment_fit.csv`)

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| LogisticRegression | yes | yes | no | no |
| RandomForest | yes | yes | no | no |

Argument:
- **Cloud / Edge:** both models are far under every limit (latency ≤ 0.92 ms vs 50 ms for Edge, size ≤ 284 KB vs 50 MB, ~266 MiB RSS fits the 1 GB Edge budget).
- **Mobile:** latency (≤ 0.92 ms vs 20 ms) and size (≤ 284 KB vs 10 MB) pass, but peak RSS of ~266 MiB exceeds the 256 MB budget. The overflow comes from the Python + NumPy/scikit-learn runtime, not the model itself (model delta is only 1.3–2.6 MiB).
- **TinyML:** RandomForest fails on size alone (284 KB > 100 KB). LogisticRegression is only 1 KB and 0.02 ms, so the model would fit, but the ~265 MiB runtime is about 1000× the 256 KB budget. It would need to be exported as 30 weights + bias in C to run on a microcontroller.

## 4. Conclusions

1. Accuracy is almost the same, but RandomForest is 47× slower per prediction and 276× larger on disk, so the linear model gives slightly better quality at a much lower cost.
2. Training is the opposite: LogisticRegression trains 6× slower because features are unscaled and lbfgs hits `max_iter=1000`. Training cost depends on preprocessing, not only on model type; adding `StandardScaler` would cut it sharply.
3. Memory measured in Python is dominated by the interpreter and libraries. This runtime overhead alone pushes both models out of the Mobile and TinyML tiers, so deployment decisions depend on the runtime, not just the saved model file.