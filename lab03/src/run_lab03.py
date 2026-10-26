"""Lab 3: params, FLOPs (hand vs thop), latency, training, trade-off plots."""
import io
import random
import time
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from thop import profile
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision import datasets, transforms

from hand_flops import hand_rows
from models import build_models

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
RES = Path(__file__).resolve().parents[1] / "results"
RES.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"

BATCH = 128
EPOCHS = 5
LR = 1e-3
LAT_RUNS = {1: 100, 128: 20}


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def get_loaders():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.2860,), (0.3530,))])
    full = datasets.FashionMNIST(DATA, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(DATA, train=False, download=True, transform=tf)
    train, val = random_split(full, [55000, 5000],
                              generator=torch.Generator().manual_seed(SEED))
    return (DataLoader(train, BATCH, shuffle=True),
            DataLoader(val, BATCH),
            DataLoader(test, BATCH))


def count_params(name, model, log):
    s = summary(model, input_size=(1, 1, 28, 28), verbose=0,
                col_names=("output_size", "num_params"))
    (RES / f"torchinfo_{name}.txt").write_text(str(s), encoding="utf-8")
    log.append(f"{name}: {s.total_params:,} params (torchinfo)")
    return s.total_params


def thop_flops(model):
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    with redirect_stdout(io.StringIO()):          # silence thop prints
        macs, _ = profile(model, inputs=(x,), verbose=False)
    return int(macs * 2)                          # thop returns MACs; 1 MAC = 2 FLOPs


@torch.no_grad()
def latency_ms(model, batch, runs):
    model.eval()
    x = torch.randn(batch, 1, 28, 28)
    for _ in range(5):                            # warm-up
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train(name, model, train_dl, val_dl, log):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        model.train()
        t0, run_loss, n = time.perf_counter(), 0.0, 0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        val_acc = evaluate(model, val_dl)
        msg = (f"{name:<10} epoch {ep + 1}/{EPOCHS} loss={run_loss / n:.4f} "
               f"val_acc={val_acc:.4f} ({time.perf_counter() - t0:.1f}s)")
        print(msg)
        log.append(msg)


def plot_tradeoffs(df):
    colors = {"MLP": "tab:blue", "CNN": "tab:orange", "Attention": "tab:green"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, col, title in zip(
        axes,
        ["params", "latency_b1_ms", "latency_b128_ms"],
        ["Parameters", "Latency, batch 1 (ms)", "Latency, batch 128 (ms)"],
    ):
        for _, r in df.iterrows():
            ax.scatter(r[col], r["test_acc"], s=120, color=colors[r["model"]],
                       label=r["model"], zorder=3)
            ax.annotate(r["model"], (r[col], r["test_acc"]),
                        textcoords="offset points", xytext=(8, 6))
        ax.set_xlabel(title)
        ax.grid(alpha=0.3)
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Test accuracy")
    fig.suptitle("Accuracy vs cost (Fashion-MNIST, 5 epochs)")
    fig.tight_layout()
    fig.savefig(RES / "tradeoff.png", dpi=150)


def main():
    set_seeds()
    log = [f"torch {torch.__version__}, threads {torch.get_num_threads()}"]
    train_dl, val_dl, test_dl = get_loaders()
    models = build_models()

    # cost: params, FLOPs, latency (before training, weights don't affect cost)
    hand_layers, hand_totals, attn_parts = hand_rows()
    rows = []
    for name, m in models.items():
        params = count_params(name, m, log)
        rows.append({
            "model": name,
            "params": params,
            "hand_flops": hand_totals[name],
            "thop_flops": thop_flops(m),
            "latency_b1_ms": round(latency_ms(m, 1, LAT_RUNS[1]), 4),
            "latency_b128_ms": round(latency_ms(m, 128, LAT_RUNS[128]), 3),
        })

    # train
    for name, m in models.items():
        set_seeds()
        train(name, m, train_dl, val_dl, log)
    for r in rows:
        r["test_acc"] = round(evaluate(models[r["model"]], test_dl), 4)

    df = pd.DataFrame(rows)
    df["thop_over_hand"] = (df["thop_flops"] / df["hand_flops"]).round(3)
    df["latency_per_img_b128_ms"] = (df["latency_b128_ms"] / 128).round(4)
    df.to_csv(RES / "results.csv", index=False)
    pd.DataFrame(hand_layers).to_csv(RES / "hand_flops_layers.csv", index=False)
    plot_tradeoffs(df)

    log += ["", "Hand FLOPs, one layer of each type:",
            pd.DataFrame(hand_layers).to_string(index=False),
            "", f"Attention layer breakdown: {attn_parts}",
            "", df.to_string(index=False)]
    text = "\n".join(log)
    (RES / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
