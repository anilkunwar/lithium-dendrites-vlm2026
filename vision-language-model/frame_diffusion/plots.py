#!/usr/bin/env python3
"""Training-time figures. Run directly to redraw the curves of a run:

python frame_diffusion/plots.py runs/fd_k4
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import CHANNELS


def read_log(run_dir):
    rows = {}
    with open(os.path.join(run_dir, "train_log.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["step"]] = {**rows.get(r["step"], {}), **r}  # a resumed run overwrites old steps
    return [rows[s] for s in sorted(rows)]


def plot_train_log(run_dir):
    """Write <run>/train_curves.png: training loss and validation MSE against step."""
    rows = read_log(run_dir)
    steps = np.array([r["step"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    val = [(r["step"], r["val_mse"]) for r in rows if "val_mse" in r]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    ax.plot(steps, loss, color="0.75", lw=0.8, label="loss (every 50 steps)")
    if len(loss) >= 20:
        w = 20
        ax.plot(steps[w - 1:], np.convolve(loss, np.ones(w) / w, mode="valid"), color="C0", lw=1.6,
                label=f"moving average ({w} points)")
    ax.set_yscale("log"), ax.set_xlabel("step"), ax.set_title("training loss (v-prediction MSE)")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)

    ax = axes[1]
    if val:
        vs, vm = map(np.array, zip(*val))
        b = int(np.argmin(vm))
        ax.plot(vs, vm, "o-", color="C1", ms=3.5, lw=1.4, label="val_mse")
        ax.plot(vs[b], vm[b], "*", color="C3", ms=14, label=f"best {vm[b]:.4g} @ step {vs[b]}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no validation yet", ha="center", va="center", transform=ax.transAxes)
    ax.set_yscale("log"), ax.set_xlabel("step")
    ax.set_title("validation: one-step prediction MSE (normalized)"), ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(run_dir, "train_curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_val_samples(last_ctx, target, pred, labels, path, title):
    """One row per validation window: last context eta, then GT / prediction / |error| per channel.

    Arrays are (N, C, H, W) in normalized units ([-1, 1]).
    """
    n = len(target)
    cols = [("last context\neta", None)] + [(f"{c}\n{k}", (i, k)) for i, c in enumerate(CHANNELS)
                                            for k in ("GT", "pred", "|err|")]
    fig, axes = plt.subplots(n, len(cols), figsize=(1.55 * len(cols), 1.65 * n + 0.6), squeeze=False)
    for r in range(n):
        for j, (name, spec) in enumerate(cols):
            ax = axes[r, j]
            if spec is None:
                ax.imshow(last_ctx[r, 0], origin="lower", cmap="viridis", vmin=-1, vmax=1)
            else:
                c, kind = spec
                if kind == "|err|":
                    ax.imshow(np.abs(pred[r, c] - target[r, c]), origin="lower", cmap="magma", vmin=0, vmax=0.5)
                else:
                    ax.imshow((target if kind == "GT" else pred)[r, c], origin="lower", cmap="viridis",
                              vmin=-1, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=8)
            if j == 0:
                ax.set_ylabel(labels[r], fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(plot_train_log(sys.argv[1]))
