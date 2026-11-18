"""Plots for training curves and image comparisons.

python plots.py runs/<name>        # redraw <run>/training_curves.png from the CSV logs
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows]) for k in rows[0]}


def smooth(y, frac=0.02):
    """Exponential moving average with a window of ~frac of the curve length."""
    if len(y) < 3:
        return y
    a = 1.0 / max(1.0, frac * len(y))
    out, m = np.empty_like(y), y[0]
    for i, v in enumerate(y):
        m = (1 - a) * m + a * v
        out[i] = m
    return out


def plot_training(run_dir):
    tr = read_csv(os.path.join(run_dir, "metrics_train.csv"))
    ev = read_csv(os.path.join(run_dir, "metrics_eval.csv"))
    if not tr:
        return None
    variables = [k[5:] for k in ev if k.startswith("rmse_")] if ev else []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    ax = axes[0, 0]
    ax.plot(tr["step"], tr["loss"], color="0.75", lw=0.8, label="train (per log interval)")
    ax.plot(tr["step"], smooth(tr["loss"]), color="C0", label="train (smoothed)")
    if ev:
        ax.plot(ev["step"], ev["val_loss"], "o-", color="C3", ms=3, label="val (EMA, fixed noise)")
        ax.plot(ev["step"], ev["train_loss_fixed"], "s--", color="C0", ms=3, label="train subset (EMA, fixed noise)")
    ax.set_yscale("log"); ax.set_title("diffusion loss"); ax.set_xlabel("step"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(tr["step"], tr["lr"], color="C2")
    ax.set_title("learning rate"); ax.set_xlabel("step")

    ax = axes[0, 2]
    ax.plot(tr["step"], tr["grad_norm"], color="0.75", lw=0.8)
    ax.plot(tr["step"], smooth(tr["grad_norm"]), color="C4")
    ax.set_yscale("log"); ax.set_title("gradient norm (before clipping)"); ax.set_xlabel("step")

    if ev:
        ax = axes[1, 0]
        for i, v in enumerate(variables):
            ax.plot(ev["step"], ev[f"rmse_{v}"], "o-", ms=3, color=f"C{i}", label=v)
        ax.set_yscale("log"); ax.set_title("sample RGB RMSE vs simulation (val)"); ax.set_xlabel("step")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        for i, v in enumerate(variables):
            ax.plot(ev["step"], ev[f"psnr_{v}"], "o-", ms=3, color=f"C{i}", label=v)
        ax.set_title("sample PSNR [dB] (val)"); ax.set_xlabel("step"); ax.legend(fontsize=8)

        ax = axes[1, 2]
        if "eta_iou" in ev:
            ax.plot(ev["step"], ev["eta_iou"], "o-", ms=3, color="C1", label="deposit IoU (eta >= 0.5)")
            ax.plot(ev["step"], ev["front_err"], "s-", ms=3, color="C5", label="front error / width")
            ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
        ax.set_title("deposit shape (val, eta decoded from jet)"); ax.set_xlabel("step")
    else:
        for ax in axes[1]:
            ax.axis("off")

    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(run_dir, "training_curves.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def rgb(x, k):
    """(3V, H, W) array -> (H, W, 3) image of variable k."""
    return np.clip(x[3 * k:3 * k + 3].transpose(1, 2, 0), 0, 1)


def plot_comparison(path, gen, variables, ref=None, titles=None):
    """Grid with one row per sample: generated (and simulated, |difference|) image of each variable.

    gen, ref: (N, 3V, H, W) numpy RGB in [0, 1].
    """
    n = gen.shape[0]
    cols = []
    for k, v in enumerate(variables):
        cols.append(("gen", k, v))
        if ref is not None:
            cols += [("sim", k, v), ("|diff|", k, v)]
    fig, axes = plt.subplots(n, len(cols), figsize=(2.2 * len(cols), 2.3 * n), squeeze=False)
    for r in range(n):
        for j, (kind, k, v) in enumerate(cols):
            ax = axes[r, j]
            if kind == "|diff|":
                im = ax.imshow(np.abs(rgb(gen[r], k) - rgb(ref[r], k)).mean(axis=-1), cmap="magma", vmin=0)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            else:
                ax.imshow(rgb(gen[r] if kind == "gen" else ref[r], k))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{v} {kind}", fontsize=9)
            if j == 0 and titles is not None:
                ax.set_ylabel(titles[r], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_evolution(path, times, gen, ref=None, k=0, name=""):
    """Variable k over time: one column per time, rows generated / simulated.

    gen: list of (3V, H, W) arrays; ref: list of arrays or None entries, or None.
    """
    rows = 1 if ref is None else 2
    fig, axes = plt.subplots(rows, len(times), figsize=(1.8 * len(times), 1.9 * rows), squeeze=False)
    for j, t in enumerate(times):
        for r in range(rows):
            img = gen[j] if r == 0 else ref[j]
            ax = axes[r, j]
            ax.set_xticks([]); ax.set_yticks([])
            if img is not None:
                ax.imshow(rgb(img, k))
            if r == 0:
                ax.set_title(f"t={t:g}", fontsize=8)
    axes[0, 0].set_ylabel(f"{name} generated", fontsize=8)
    if rows == 2:
        axes[1, 0].set_ylabel(f"{name} simulated", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    print(plot_training(p.parse_args().run_dir))
