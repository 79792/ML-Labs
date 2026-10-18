"""Lab 1: baselines, system cost measurements, deployment fit."""
import os
import random
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from memory_profiler import memory_usage
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.exceptions import ConvergenceWarning

# LogReg on unscaled data hits max_iter; params are fixed by the manual
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 42
RES = Path(__file__).resolve().parents[1] / "results"
RES.mkdir(parents=True, exist_ok=True)

N_TRAIN_RUNS = 5
N_INFER_RUNS = 100

# Budgets: (min_mem_MB, max_mem_MB, max_latency_ms, max_size_MB)
BUDGETS = {
    "Cloud": (1024, None, 100, 500),
    "Edge": (256, 1024, 50, 50),
    "Mobile": (64, 256, 20, 10),
    "TinyML": (0, 0.25, 10, 0.1),
}


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    except ImportError:
        pass


def make_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }


def time_training(model, X, y):
    # 1 warm-up + N timed runs, fresh clone each time
    clone(model).fit(X, y)
    times = []
    for _ in range(N_TRAIN_RUNS):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X, y)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_inference(model, x):
    # single sample, 1 warm-up + 100 timed runs
    model.predict(x)
    times = []
    for _ in range(N_INFER_RUNS):
        t0 = time.perf_counter()
        model.predict(x)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def peak_mem(fn, *args):
    # peak process RSS (MiB) while fn runs
    return float(memory_usage((fn, args), interval=0.001, max_usage=True))


def fits(budget, mem_mb, lat_ms, size_mb):
    lo, hi, max_lat, max_size = budget
    # device must provide at least what the model needs
    mem_ok = mem_mb <= (hi if hi is not None else float("inf"))
    return mem_ok and lat_ms <= max_lat and size_mb <= max_size


def main():
    set_seeds()
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    x_one = X_te[:1]
    proc = psutil.Process(os.getpid())
    base_rss = proc.memory_info().rss / 2**20

    acc_rows, cost_rows, fitted = [], [], {}
    for name, model in make_models().items():
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        acc_rows.append({"model": name, "test_accuracy": round(acc, 4)})

        train_s = time_training(model, X_tr, y_tr)
        infer_s = time_inference(model, x_one)

        path = RES / f"model_{name}.joblib"
        joblib.dump(model, path)
        size_b = path.stat().st_size

        train_peak = peak_mem(clone(model).fit, X_tr, y_tr)
        infer_peak = peak_mem(model.predict, x_one)

        cost_rows.append({
            "model": name,
            "train_time_ms_median": round(train_s * 1e3, 3),
            "infer_latency_ms_median": round(infer_s * 1e3, 4),
            "model_size_bytes": size_b,
            "model_size_KB": round(size_b / 1024, 2),
            "peak_rss_train_MiB": round(train_peak, 1),
            "peak_rss_infer_MiB": round(infer_peak, 1),
            "peak_delta_train_MiB": round(train_peak - base_rss, 1),
        })

        fitted[name] = model

    # manual names results/model.joblib: store both models in one file
    joblib.dump(fitted, RES / "model.joblib")

    acc_df = pd.DataFrame(acc_rows)
    cost_df = pd.DataFrame(cost_rows)
    acc_df.to_csv(RES / "baseline_accuracy.csv", index=False)
    cost_df.to_csv(RES / "system_cost.csv", index=False)

    # deployment fit uses inference peak RSS as memory need
    fit_rows = []
    for r in cost_rows:
        row = {"model": r["model"]}
        for tier, b in BUDGETS.items():
            row[tier] = "yes" if fits(
                b, r["peak_rss_infer_MiB"], r["infer_latency_ms_median"],
                r["model_size_bytes"] / 2**20
            ) else "no"
        fit_rows.append(row)
    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(RES / "deployment_fit.csv", index=False)

    print(acc_df.to_string(index=False), "\n")
    print(cost_df.to_string(index=False), "\n")
    print(fit_df.to_string(index=False))


if __name__ == "__main__":
    main()
