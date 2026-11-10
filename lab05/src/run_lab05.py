"""Lab 5: own autograd vs PyTorch, make_moons training, eager vs compiled, contiguity.

Run from repo root:  python lab05/src/run_lab05.py
"""
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_moons
from torch import nn

from engine import Value, vsum

SEED = 42
RES = Path(__file__).resolve().parents[1] / "results"
RES.mkdir(parents=True, exist_ok=True)
LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


def median_time(fn, runs, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------- 4. gradient check ----------
def numerical_grad(f, values, i, eps=1e-6):
    old = values[i]
    values[i] = old + eps
    lp = f(values)
    values[i] = old - eps
    lm = f(values)
    values[i] = old
    return (lp - lm) / (2 * eps)


def grad_check():
    log("=== 4. Gradient check: ((a*b + c)**2).relu().sum() ===")
    rng = np.random.default_rng(SEED)
    n = 5
    a0, b0, c0 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)

    # own engine
    a = [Value(x) for x in a0]
    b = [Value(x) for x in b0]
    c = [Value(x) for x in c0]
    out = vsum([((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(a, b, c)])
    out.backward()
    mine = np.array([v.grad for v in a + b + c])

    # PyTorch autograd, float64 for a fair comparison
    ta = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b0, dtype=torch.float64, requires_grad=True)
    tc = torch.tensor(c0, dtype=torch.float64, requires_grad=True)
    tout = ((ta * tb + tc) ** 2).relu().sum()
    tout.backward()
    torch_g = torch.cat([ta.grad, tb.grad, tc.grad]).numpy()

    # numerical central difference
    flat = list(np.concatenate([a0, b0, c0]))

    def f(v):
        aa, bb, cc = np.array(v[:n]), np.array(v[n:2 * n]), np.array(v[2 * n:])
        return float(np.maximum((aa * bb + cc) ** 2, 0).sum())
    num = np.array([numerical_grad(f, flat, i) for i in range(3 * n)])

    names = [f"{p}[{i}]" for p in "abc" for i in range(n)]
    df = pd.DataFrame({"param": names, "engine": mine, "pytorch": torch_g,
                       "numerical": num,
                       "abs_diff_torch": np.abs(mine - torch_g),
                       "abs_diff_num": np.abs(mine - num)})
    df.to_csv(RES / "grad_check.csv", index=False)
    log(f"Forward: engine {out.data:.10f}, pytorch {tout.item():.10f}")
    log(f"Max |engine - pytorch|   = {df.abs_diff_torch.max():.3e}")
    log(f"Max |engine - numerical| = {df.abs_diff_num.max():.3e}")
    ok = df.abs_diff_torch.max() < 1e-6 and df.abs_diff_num.max() < 1e-6
    log(f"PASS (< 1e-6): {ok}")
    log(df.to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    return df


# ---------- 5. make_moons training ----------
EPOCHS, LR, N = 100, 0.5, 200


def init_weights():
    rng = np.random.default_rng(SEED)
    return {"W1": rng.normal(0, 0.5, (2, 8)), "b1": np.zeros(8),
            "W2": rng.normal(0, 0.5, (8, 1)), "b2": np.zeros(1)}


def train_engine(X, y, w0):
    W1 = [[Value(w0["W1"][i, j]) for j in range(8)] for i in range(2)]
    b1 = [Value(v) for v in w0["b1"]]
    W2 = [Value(w0["W2"][j, 0]) for j in range(8)]
    b2 = Value(w0["b2"][0])
    params = [w for row in W1 for w in row] + b1 + W2 + [b2]
    eps = 1e-7

    for _ in range(EPOCHS):
        losses = []
        for x, t in zip(X, y):
            h = [(W1[0][j] * x[0] + W1[1][j] * x[1] + b1[j]).relu() for j in range(8)]
            p = (vsum([W2[j] * h[j] for j in range(8)]) + b2).sigmoid()
            # BCE, eps keeps log away from 0
            losses.append(-(t * (p + eps).log() + (1 - t) * (1 - p + eps).log()))
        loss = vsum(losses) * (1.0 / len(X))
        loss.backward()
        for prm in params:
            prm.data -= LR * prm.grad
    preds = []
    for x in X:
        h = [max(0.0, W1[0][j].data * x[0] + W1[1][j].data * x[1] + b1[j].data) for j in range(8)]
        z = sum(W2[j].data * h[j] for j in range(8)) + b2.data
        preds.append(1 if z > 0 else 0)
    return loss.data, float(np.mean(np.array(preds) == y))


def train_torch(X, y, w0):
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
    with torch.no_grad():                 # same init as engine
        model[0].weight.copy_(torch.tensor(w0["W1"].T))
        model[0].bias.copy_(torch.tensor(w0["b1"]))
        model[2].weight.copy_(torch.tensor(w0["W2"].T))
        model[2].bias.copy_(torch.tensor(w0["b2"]))
    model = model.double()
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64).unsqueeze(1)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()      # sigmoid + BCE in one stable op
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    acc = ((model(Xt) > 0).double() == yt).double().mean().item()
    return loss.item(), acc


def moons():
    log("\n=== 5. make_moons, 2 -> 8 -> 1, sigmoid + BCE, full-batch SGD ===")
    X, y = make_moons(n_samples=N, noise=0.1, random_state=SEED)
    w0 = init_weights()
    e_loss, e_acc = train_engine(X, y, w0)
    t_loss, t_acc = train_torch(X, y, w0)
    log(f"Engine : final loss {e_loss:.4f}, train acc {e_acc:.3f}")
    log(f"PyTorch: final loss {t_loss:.4f}, train acc {t_acc:.3f}")

    e_t = median_time(lambda: train_engine(X, y, w0), runs=5, warmup=0)
    t_t = median_time(lambda: train_torch(X, y, w0), runs=5, warmup=1)
    log(f"Training time ({EPOCHS} epochs, median of 5): engine {e_t:.3f} s, "
        f"pytorch {t_t:.4f} s, pytorch is {e_t / t_t:.0f}x faster")
    return {"engine_s": e_t, "pytorch_s": t_t, "engine_acc": e_acc, "pytorch_acc": t_acc,
            "engine_loss": e_loss, "pytorch_loss": t_loss}


# ---------- 6. graph, eager vs compiled, contiguity ----------
def print_grad_graph(t, depth=0, seen=None, lines=None):
    seen = set() if seen is None else seen
    lines = [] if lines is None else lines
    fn = t if depth else t.grad_fn
    if fn is None or id(fn) in seen:
        return lines
    seen.add(id(fn))
    lines.append("  " * depth + type(fn).__name__)
    for nxt, _ in fn.next_functions:
        if nxt is not None:
            print_grad_graph(nxt, depth + 1, seen, lines)
    return lines


def compiled_vs_eager():
    log("\n=== 6. Graph inspection, eager vs compiled, tensor layout ===")
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(),
                          nn.Linear(512, 512), nn.ReLU(),
                          nn.Linear(512, 10)).eval()
    x = torch.randn(64, 256)

    # autograd graph (backward graph built during forward)
    out = model(x).sum()
    graph = "\n".join(print_grad_graph(out))
    log("Autograd graph (grad_fn chain from output):\n" + graph)

    with torch.no_grad():
        traced = torch.jit.trace(model, x)
    (RES / "graph.txt").write_text(
        "Autograd graph:\n" + graph + "\n\nTorchScript traced graph:\n" + str(traced.graph),
        encoding="utf-8")

    # tensor inspection
    log(f"Input: shape {tuple(x.shape)}, dtype {x.dtype}, contiguous {x.is_contiguous()}, "
        f"stride {x.stride()}")
    w = model[0].weight
    log(f"Linear(256,512).weight: shape {tuple(w.shape)}, dtype {w.dtype}, "
        f"contiguous {w.is_contiguous()}, stride {w.stride()}")

    rows = []
    runs = 200

    def bench(name, m):
        with torch.no_grad():
            t = median_time(lambda: m(x), runs=runs, warmup=20)
        rows.append({"mode": name, "median_ms": round(t * 1e3, 4)})
        log(f"{name:<20} {t * 1e3:.4f} ms (median of {runs}, after warm-up)")

    bench("eager", model)
    bench("torch.jit.trace", traced)
    try:
        compiled = torch.compile(model)
        with torch.no_grad():
            compiled(x)                   # first call compiles
        bench("torch.compile", compiled)
    except Exception as e:                # e.g. no compiler on Windows
        msg = str(e).splitlines()[0][:120]
        log(f"torch.compile not available here: {type(e).__name__}: {msg}")
        rows.append({"mode": "torch.compile", "median_ms": None})

    # contiguity
    log("\nNon-contiguous view cost (4096x4096 float32):")
    big = torch.randn(4096, 4096)
    view = big.t()                        # same memory, swapped strides
    copy = view.contiguous()
    log(f"big.t(): contiguous {view.is_contiguous()}, stride {view.stride()}; "
        f".contiguous(): stride {copy.stride()}")

    tests = {
        "x + 1 (contiguous)": lambda: copy + 1,
        "x + 1 (non-contiguous view)": lambda: view + 1,
        "row sum (contiguous)": lambda: copy.sum(dim=1),
        "row sum (non-contiguous view)": lambda: view.sum(dim=1),
        "cost of .contiguous() copy": lambda: view.contiguous(),
    }
    for name, fn in tests.items():
        t = median_time(fn, runs=20, warmup=3)
        rows.append({"mode": name, "median_ms": round(t * 1e3, 3)})
        log(f"{name:<32} {t * 1e3:.3f} ms")

    pd.DataFrame(rows).to_csv(RES / "timings.csv", index=False)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    log(f"torch {torch.__version__}, threads {torch.get_num_threads()}")
    grad_check()
    m = moons()
    pd.DataFrame([m]).to_csv(RES / "moons.csv", index=False)
    compiled_vs_eager()
    (RES / "summary.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
