"""Lab 4: one-command data pipeline for UCI Adult.

Run from repo root:  python lab04/src/run_lab04.py
"""
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from pipeline import CATEGORICAL, NUMERIC, build_preprocessor, ensure_object

ROOT = Path(__file__).resolve().parents[2]
LAB = Path(__file__).resolve().parents[1]
RES = LAB / "results"
RAW = ROOT / "data" / "raw" / "adult_raw.csv"
PROCESSED = ROOT / "data" / "processed" / "adult_processed.npz"

CONFIG = {
    "dataset": {"source": "openml", "name": "adult", "version": 2},
    "seed": 42,
    "split": {"test_size": 0.3, "stratify": "income"},
    "null_policy": "'?' and NaN are missing",
    "impute": {"numeric": "median", "categorical": "most_frequent"},
    "clip_percentiles": [1, 99],
    "dedup": "exact duplicates dropped from train only",
    "derived_features": ["capital_net = capital-gain - capital-loss",
                         "hours_per_age = hours-per-week / (age - 16)"],
    "model": "LogisticRegression(max_iter=1000)",
    "positive_label": ">50K",
}

INT_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT_COLS = CATEGORICAL + ["income"]
RANGES = {"age": (17, 90), "hours-per-week": (1, 99),
          "capital-gain": (0, 99999), "capital-loss": (0, 4356)}
FEATURES = NUMERIC + CATEGORICAL


# ---------- 1. ingestion ----------
def ingest(log):
    """Download once, keep an untouched raw copy, always read from that copy."""
    if not RAW.exists():
        from sklearn.datasets import fetch_openml
        bunch = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
        RAW.parent.mkdir(parents=True, exist_ok=True)
        bunch.frame.to_csv(RAW, index=False)
        log.append(f"Downloaded from OpenML and saved raw copy to {RAW.relative_to(ROOT)}")
    raw = pd.read_csv(RAW)
    raw_hash = hashlib.sha256(RAW.read_bytes()).hexdigest()
    log.append(f"Ingestion: {raw.shape[0]} rows, {raw.shape[1]} columns")
    log.append(f"Raw file SHA256: {raw_hash}")
    if raw.shape != (48842, 15):
        log.append(f"WARNING: expected (48842, 15), got {raw.shape}")
    return raw, raw_hash


def rename_target(df):
    # OpenML names the label column 'class'
    return df.rename(columns={"class": "income"}) if "class" in df.columns else df.copy()


def is_missing(s):
    return s.isna() | (s.astype(str).str.strip() == "?")


# ---------- 2. validation ----------
def validate(df, log):
    rows = []

    # schema: types
    for c in INT_COLS:
        vals = pd.to_numeric(df[c], errors="coerce")
        bad_type = int((vals.isna() & ~is_missing(df[c])).sum())
        non_int = int(((vals.dropna() % 1) != 0).sum())
        rows.append({"check": "type int", "column": c, "violations": bad_type + non_int})
    for c in CAT_COLS:
        numeric_like = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
        rows.append({"check": "type category", "column": c, "violations": numeric_like})

    # ranges
    for c, (lo, hi) in RANGES.items():
        v = pd.to_numeric(df[c], errors="coerce")
        n = int(((v < lo) | (v > hi)).sum())
        rows.append({"check": f"range {lo}-{hi}", "column": c, "violations": n})

    # missing
    for c in df.columns:
        rows.append({"check": "missing ('?' or NaN)", "column": c,
                     "violations": int(is_missing(df[c]).sum())})

    # duplicates
    rows.append({"check": "exact duplicate rows", "column": "all",
                 "violations": int(df.duplicated().sum())})

    # label values
    labels = df["income"].astype(str).str.strip().str.rstrip(".")
    bad_label = int((~labels.isin(["<=50K", ">50K"])).sum())
    rows.append({"check": "label in {<=50K, >50K}", "column": "income", "violations": bad_label})

    rep = pd.DataFrame(rows)
    rep["status"] = np.where(rep["violations"] == 0, "OK", "VIOLATION")
    rep.to_csv(RES / "validation_report.csv", index=False)

    lines = ["# Validation report", "",
             f"Rows checked: {len(df)}. No rows were dropped at this step.", "",
             "| Check | Column | Violations | Status |", "|---|---|---|---|"]
    lines += [f"| {r.check} | {r.column} | {r.violations} | {r.status} |" for r in rep.itertuples()]
    (RES / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    bad = rep[rep["violations"] > 0]
    log.append(f"Validation: {len(rep)} checks, {len(bad)} with violations")
    log += [f"  {r.check:<24} {r.column:<16} {r.violations}" for r in bad.itertuples()]
    return rep


# ---------- 6. label quality ----------
def label_audit(df, seed, log):
    # duplicate-conflict: same 14 features, different label
    g = df.groupby(FEATURES, dropna=False)["income"].nunique()
    conflict_groups = g[g > 1]
    conflict_rows = int(df.set_index(FEATURES).index.isin(conflict_groups.index).sum())
    log.append(f"Duplicate-conflict audit: {len(conflict_groups)} feature groups, "
               f"{conflict_rows} rows with conflicting income labels")

    # 50 random rows checked against data dictionary
    s = df.sample(50, random_state=seed).copy()
    edu_map = df.groupby("education")["education-num"].nunique()
    s["flag_age_range"] = ~s["age"].between(17, 90)
    s["flag_hours_range"] = ~s["hours-per-week"].between(1, 99)
    s["flag_gain_and_loss"] = (s["capital-gain"] > 0) & (s["capital-loss"] > 0)
    s["flag_husband_female"] = (s["relationship"] == "Husband") & (s["sex"] == "Female")
    s["flag_wife_male"] = (s["relationship"] == "Wife") & (s["sex"] == "Male")
    s["flag_edu_mismatch"] = s["education"].map(edu_map).fillna(1) > 1
    s["flag_missing"] = s[FEATURES].apply(is_missing).any(axis=1)
    flag_cols = [c for c in s.columns if c.startswith("flag_")]
    s["any_flag"] = s[flag_cols].any(axis=1)
    s.to_csv(RES / "label_audit_sample.csv", index=False)
    counts = {c: int(s[c].sum()) for c in flag_cols}
    log.append(f"50-row manual audit flags: {counts}")
    return len(conflict_groups), conflict_rows, counts


# ---------- 4–5. split, clean, transform, evaluate ----------
def split_and_clean(df, seed, log):
    X = ensure_object(df[FEATURES])
    y = (df["income"].astype(str).str.strip().str.rstrip(".") == ">50K").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed)

    # drop exact duplicates (features + label) from train only
    tr = X_tr.assign(_y=y_tr.values)
    before = len(tr)
    tr = tr.drop_duplicates()
    X_tr, y_tr = tr.drop(columns="_y"), tr["_y"]
    log.append(f"Split 70/30 stratified: train {before}, test {len(X_te)}")
    log.append(f"Dropped {before - len(tr)} exact duplicates from train (test untouched); "
               f"train now {len(tr)}")
    return X_tr, X_te, y_tr, y_te


def evaluate(X_tr, X_te, y_tr, y_te, with_derived):
    pre = build_preprocessor(with_derived)
    Xt_tr = pre.fit_transform(X_tr)       # fit on train only
    Xt_te = pre.transform(X_te)
    clf = LogisticRegression(max_iter=1000).fit(Xt_tr, y_tr)
    p = clf.predict(Xt_te)
    return pre, Xt_tr, Xt_te, {
        "accuracy": round(accuracy_score(y_te, p), 4),
        "f1": round(f1_score(y_te, p), 4),
        "n_features": Xt_tr.shape[1],
    }


# ---------- 7. hash, config, data card ----------
def dataset_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def to_yaml(d, indent=0):
    """Tiny YAML writer for flat/nested dicts and lists (no extra dependency)."""
    pad, out = "  " * indent, []
    for k, v in d.items():
        if isinstance(v, dict):
            out.append(f"{pad}{k}:")
            out.append(to_yaml(v, indent + 1))
        elif isinstance(v, list):
            out.append(f"{pad}{k}:")
            out += [f"{pad}  - {json.dumps(i)}" for i in v]
        else:
            out.append(f"{pad}{k}: {json.dumps(v)}")
    return "\n".join(out)


def write_data_card(df, raw_hash, proc_hash, val, audit, metrics):
    pos = (df["income"].astype(str).str.rstrip(".") == ">50K").mean()
    sex = df["sex"].value_counts(normalize=True).round(3).to_dict()
    race = df["race"].value_counts(normalize=True).round(3).to_dict()
    missing = {c: int(is_missing(df[c]).sum()) for c in df.columns if is_missing(df[c]).sum()}
    text = f"""# Data card: UCI Adult (Census Income)

## Source
- UCI Machine Learning Repository, "Adult" dataset (Becker & Kohavi, 1996), extracted from the 1994 US Census database.
- Loaded with `sklearn.datasets.fetch_openml(name='adult', version=2)` (OpenML ID 1590).
- License: Creative Commons Attribution 4.0 (CC BY 4.0).

## Size and structure
- {len(df)} rows, {df.shape[1]} columns: 14 features (6 numeric, 8 categorical) + label `income` (`<=50K` / `>50K`).
- Class balance: {pos:.1%} `>50K`, {1 - pos:.1%} `<=50K` (imbalanced, so F1 is reported next to accuracy).
- Missing values (`?`/NaN): {missing}.
- Exact duplicate rows: {int(df.duplicated().sum())}. Feature-duplicate groups with conflicting labels: {audit[0]} ({audit[1]} rows).

## Processing
- 70/30 stratified split (seed 42) before any cleaning. Imputer (median / mode), 1st–99th percentile clipping and scaling are fitted on train only. Duplicates dropped from train only.
- Derived features: `capital_net`, `hours_per_age`.
- Raw file SHA256: `{raw_hash}`
- Processed dataset SHA256: `{proc_hash}`

## Known biases
- Data is from 1994; incomes are not adjusted for inflation, so the 50K threshold means something different today.
- Sex is imbalanced: {sex}.
- Race is very imbalanced: {race}. A model can be much less reliable for small groups.
- `native-country` is mostly United-States, other countries have very few rows.
- `fnlwgt` is a census sampling weight, not a personal attribute.
- The label is binary and self-reported; models trained on it can learn historical pay gaps (for example by sex or race).

## Privacy concerns
- No names or IDs, but it has sensitive attributes: sex, race, marital status, relationship, native country.
- Rare combinations (for example age + country + occupation) could make some people re-identifiable.
- Sensitive attributes should not be used for real decisions about people (credit, hiring) without a fairness review.

## Intended use
- Teaching and benchmarking tabular ML pipelines. Not for real income or credit decisions.
- Baseline (LogisticRegression): accuracy {metrics['accuracy']}, F1 {metrics['f1']} with derived features.
"""
    (LAB / "data_card.md").write_text(text, encoding="utf-8")


def main():
    RES.mkdir(parents=True, exist_ok=True)
    seed = CONFIG["seed"]
    log = []

    raw, raw_hash = ingest(log)
    df = rename_target(raw)        # copy with readable label name; raw stays untouched

    val = validate(df, log)
    audit = label_audit(df, seed, log)

    X_tr, X_te, y_tr, y_te = split_and_clean(df, seed, log)
    _, _, _, m_base = evaluate(X_tr, X_te, y_tr, y_te, with_derived=False)
    pre, Xt_tr, Xt_te, m_der = evaluate(X_tr, X_te, y_tr, y_te, with_derived=True)
    joblib.dump(pre, RES / "preprocessor.joblib")

    met = pd.DataFrame([
        {"features": "14 original", **m_base},
        {"features": "14 + capital_net + hours_per_age", **m_der},
    ])
    met.to_csv(RES / "metrics.csv", index=False)
    log += ["", "LogisticRegression baseline on test:", met.to_string(index=False)]

    proc_hash = dataset_hash(Xt_tr, y_tr.values, Xt_te, y_te.values)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED, X_train=Xt_tr, y_train=y_tr.values,
                        X_test=Xt_te, y_test=y_te.values)
    log.append(f"\nProcessed dataset SHA256: {proc_hash}")

    cfg = {**CONFIG,
           "hashes": {"raw_sha256": raw_hash, "processed_sha256": proc_hash},
           "environment": {"python": platform.python_version(),
                           "sklearn": sklearn.__version__, "pandas": pd.__version__}}
    (RES / "config.yaml").write_text(to_yaml(cfg) + "\n", encoding="utf-8")

    write_data_card(df, raw_hash, proc_hash, val, audit, m_der)
    text = "\n".join(log)
    (RES / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
