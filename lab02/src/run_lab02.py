"""Lab 2: checks, gradient check, training sweep, timing."""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from nn import accuracy, backward, cross_entropy, forward, init_params, sgd_step

SEED = 42
RES = Path(__file__).resolve().parents[1] / "results"
RES.mkdir(parents=True, exist_ok=True)

LRS = [0.001, 0.01, 0.1]
WIDTHS = [16, 64]
EPOCHS = 50
BATCH = 32
N_TIMING = 100


def load_data():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0                      # pixels 0..16 -> 0..1
    Y = np.eye(10)[y]                 # one-hot
    return train_test_split(X, Y, test_size=0.3, stratify=y, random_state=SEED)


def check_forward_and_loss(X, Y, log):
    p = init_params(16)
    probs, _ = forward(p, X)
    assert probs.shape == (X.shape[0], 10)
    max_dev = float(np.abs(probs.sum(1) - 1).max())
    assert max_dev < 1e-6
    loss = cross_entropy(probs, Y)
    log.append(f"Forward output shape: {probs.shape}")
    log.append(f"Max |row sum - 1|: {max_dev:.2e}")
    log.append(f"Untrained loss: {loss:.4f} (ln10 = {np.log(10):.4f})")


def grad_check(X, Y, log, std, tag, width=16, n_checks=10, eps=1e-5):
    rng = np.random.default_rng(SEED)
    p = init_params(width, std=std)
    Xb, Yb = X[:32], Y[:32]
    _, cache = forward(p, Xb)
    grads = backward(p, cache, Yb)

    rows, passed = [], 0
    names = ["W1", "W2", "W3"]
    for i in range(n_checks):
        name = names[i % 3]
        # sample only weights with non-zero gradient (zero ones pass trivially)
        nz = np.argwhere(grads[name] != 0)
        idx = tuple(nz[rng.integers(0, len(nz))])
        old = p[name][idx]
        p[name][idx] = old + eps
        lp = cross_entropy(forward(p, Xb)[0], Yb)
        p[name][idx] = old - eps
        lm = cross_entropy(forward(p, Xb)[0], Yb)
        p[name][idx] = old
        num = (lp - lm) / (2 * eps)
        ana = grads[name][idx]
        denom = max(abs(num) + abs(ana), 1e-12)
        rel = abs(num - ana) / denom
        ok = rel < 1e-6
        passed += ok
        rows.append({"param": name, "index": tuple(int(j) for j in idx),
                     "analytic": ana, "numerical": num,
                     "rel_error": rel, "pass": bool(ok)})

    df = pd.DataFrame(rows)
    df.to_csv(RES / f"grad_check_{tag}.csv", index=False)
    log.append(f"\nGradient check (init std={std}): {passed}/{n_checks} passed (rel error < 1e-6)")
    log.append(df.to_string(index=False))
    return passed


def train(width, lr, X_tr, Y_tr, X_te, Y_te):
    rng = np.random.default_rng(SEED)
    p = init_params(width)
    losses = []
    n = X_tr.shape[0]
    for ep in range(EPOCHS):
        perm = rng.permutation(n)
        ep_loss = 0.0
        for s in range(0, n, BATCH):
            b = perm[s:s + BATCH]
            probs, cache = forward(p, X_tr[b])
            ep_loss += cross_entropy(probs, Y_tr[b]) * len(b)
            sgd_step(p, backward(p, cache, Y_tr[b]), lr)
        losses.append(ep_loss / n)
        print(f"W={width:<3} lr={lr:<6} epoch {ep + 1:>2}/{EPOCHS} loss={losses[-1]:.4f}")
    return p, losses, accuracy(p, X_te, Y_te)


def time_passes(width, X, Y):
    p = init_params(width)
    Xb, Yb = X[:BATCH], Y[:BATCH]
    _, cache = forward(p, Xb)            # warm-up
    backward(p, cache, Yb)
    f_t, b_t = [], []
    for _ in range(N_TIMING):
        t0 = time.perf_counter()
        _, cache = forward(p, Xb)
        f_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        backward(p, cache, Yb)
        b_t.append(time.perf_counter() - t0)
    return float(np.median(f_t)) * 1e6, float(np.median(b_t)) * 1e6


def main():
    np.random.seed(SEED)
    X_tr, X_te, Y_tr, Y_te = load_data()
    log = [f"Train: {X_tr.shape}, Test: {X_te.shape}"]

    check_forward_and_loss(X_tr, Y_tr, log)
    # at std=0.01 grads are ~1e-6, so float round-off dominates; std=0.1 is the main check
    grad_check(X_tr, Y_tr, log, std=0.1, tag="main")
    grad_check(X_tr, Y_tr, log, std=0.01, tag="std001")

    # training sweep
    curves, acc_rows = {}, []
    for w in WIDTHS:
        for lr in LRS:
            _, losses, acc = train(w, lr, X_tr, Y_tr, X_te, Y_te)
            curves[(w, lr)] = losses
            acc_rows.append({"width": w, "lr": lr,
                             "final_train_loss": round(losses[-1], 4),
                             "test_accuracy": round(acc, 4)})
    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(RES / "accuracy.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, w in zip(axes, WIDTHS):
        # distinct line styles: lr=0.001 and 0.01 overlap at ln10
        for lr, ls in zip(LRS, [":", "--", "-"]):
            ax.plot(range(1, EPOCHS + 1), curves[(w, lr)], ls, lw=2, label=f"lr={lr}")
        ax.set_title(f"Hidden width W={w}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Training cross-entropy loss")
    fig.tight_layout()
    fig.savefig(RES / "loss_curves.png", dpi=150)

    # timing, batch of 32
    t_rows = []
    for w in WIDTHS:
        f_us, b_us = time_passes(w, X_tr, Y_tr)
        t_rows.append({"width": w, "forward_us_median": round(f_us, 2),
                       "backward_us_median": round(b_us, 2),
                       "backward_over_forward": round(b_us / f_us, 2)})
    t_df = pd.DataFrame(t_rows)
    t_df.to_csv(RES / "timing.csv", index=False)

    log += ["", acc_df.to_string(index=False), "", t_df.to_string(index=False)]
    text = "\n".join(log)
    (RES / "summary.txt").write_text(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
