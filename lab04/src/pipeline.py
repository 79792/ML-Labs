"""Reusable transformers. Kept in a module so joblib can load them later."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex", "native-country"]
DERIVED = ["capital_net", "hours_per_age"]


def to_missing(df):
    """Null policy: '?' means missing. Row-wise, so no leakage."""
    return df.replace("?", np.nan)


def add_derived(df):
    """Row-wise derived features, no statistics from data."""
    df = df.copy()
    df["capital_net"] = df["capital-gain"] - df["capital-loss"]
    df["hours_per_age"] = df["hours-per-week"] / (df["age"] - 16)
    return df


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to [low, high] percentiles learned on train."""

    def __init__(self, low=1, high=99):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, self.low, axis=0)
        self.hi_ = np.nanpercentile(X, self.high, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def build_preprocessor(with_derived):
    num_cols = NUMERIC + (DERIVED if with_derived else [])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", PercentileClipper(1, 99)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    cols = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, CATEGORICAL),
    ])
    steps = [("missing", FunctionTransformer(to_missing))]
    if with_derived:
        steps.append(("derive", FunctionTransformer(add_derived)))
    steps.append(("columns", cols))
    return Pipeline(steps)


def ensure_object(df):
    """Categories as plain strings so '?' replace and imputing behave the same."""
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype("object")
    return df


__all__ = ["NUMERIC", "CATEGORICAL", "DERIVED", "build_preprocessor",
           "PercentileClipper", "to_missing", "add_derived", "ensure_object"]
