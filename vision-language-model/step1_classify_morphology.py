#!/usr/bin/env python3
"""Classify each case as uniform deposition or dendritic growth from its eta images.

Reads data/<case>/image_files/eta/eta_t<time>.jpg (jet colormap, written by step0), maps the
colors back to eta, and measures the deposit front attached to the left boundary in every frame:

- amplitude: spread of the front position across rows (95th - 5th percentile), / domain width
- length ratio: length of the eta = 0.5 contour / domain height (1.0 for a flat front)

The last frame decides the class:
  uniform       amplitude < --uniform-amp  and length ratio < --uniform-len
  dendritic     amplitude >= --dendrite-amp or length ratio >= --dendrite-len
  transitional  anything in between (wavy front, not yet dendrites)
The onset time is the first frame whose front is no longer uniform.

python step1_classify_morphology.py
python step1_classify_morphology.py --data data --output data/summary/morphology --every 2
"""

import argparse
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

CLASSES = ("uniform", "transitional", "dendritic")
COLORS = {"uniform": "#2b7bba", "transitional": "#e0a526", "dendritic": "#c8372d", "error": "#888888"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="data/summary/morphology")
    p.add_argument("--size", type=int, default=300, help="images are downsampled to this size for analysis")
    p.add_argument("--every", type=int, default=1, help="analyze every n-th frame (the last frame always)")
    p.add_argument("--threshold", type=float, default=0.5, help="eta threshold for the deposit")
    p.add_argument("--uniform-amp", type=float, default=0.03)
    p.add_argument("--uniform-len", type=float, default=1.10)
    p.add_argument("--dendrite-amp", type=float, default=0.10)
    p.add_argument("--dendrite-len", type=float, default=1.50)
    p.add_argument("--short-travel", type=float, default=0.10,
                   help="flag cases whose front moved less than this fraction of the width")
    return p.parse_args()


def jet_lookup(levels=64):
    """Table mapping quantized RGB to the jet value of the nearest jet color, plus that color distance."""
    jet = cm.get_cmap("jet")(np.linspace(0, 1, 1024))[:, :3] * 255
    q = (np.arange(levels) + 0.5) * 256 / levels
    grid = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1).reshape(-1, 3)
    dist, idx = cKDTree(jet).query(grid)
    return (idx / 1023).astype(np.float32), dist.astype(np.float32), levels


def decode(path, lut, size):
    """Image -> eta in [0, 1] (row 0 = top of the domain) and the fraction of non-jet pixels."""
    values, dist, levels = lut
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    rgb = np.asarray(im).astype(np.int32) * levels // 256
    flat = (rgb[..., 0] * levels + rgb[..., 1]) * levels + rgb[..., 2]
    return values[flat], float((dist[flat] > 60).mean())


def front_metrics(eta, threshold):
    mask = eta >= threshold
    labels, _ = ndimage.label(mask)  # 4-connectivity
    left = np.setdiff1d(np.unique(labels[:, 0]), [0])
    attached = np.isin(labels, left)
    H, W = eta.shape
    rows = attached.any(axis=1)
    if not rows.any():
        return None
    # Rightmost attached deposit pixel in each row; rows without deposit count as 0.
    front = np.where(rows, W - 1 - np.argmax(attached[:, ::-1], axis=1), 0) / (W - 1)
    contours = measure.find_contours(eta, threshold)
    length = sum(np.sum(np.hypot(*np.diff(c, axis=0).T)) for c in contours) / H
    return {
        "front_mean": float(front.mean()),
        "front_max": float(front.max()),
        "amplitude": float(np.percentile(front, 95) - np.percentile(front, 5)),
        "roughness": float(front.std()),
        "length_ratio": float(length),
        "detached_area": float((mask & ~attached).mean()),
    }


def classify(m, a):
    if m["amplitude"] >= a.dendrite_amp or m["length_ratio"] >= a.dendrite_len:
        return "dendritic"
    if m["amplitude"] < a.uniform_amp and m["length_ratio"] < a.uniform_len:
        return "uniform"
    return "transitional"


def frame_files(case_dir):
    files = glob.glob(os.path.join(case_dir, "image_files", "eta", "eta_t*.jpg"))
    files += glob.glob(os.path.join(case_dir, "image_files", "eta", "eta_t*.png"))
    timed = []
    for f in files:
        m = re.search(r"eta_t([-+0-9.eE]+)\.(?:jpg|png)$", os.path.basename(f))
        if m:
            timed.append((float(m.group(1)), f))
    return sorted(timed)


def analyze_case(case_dir, lut, a):
    frames = frame_files(case_dir)
    if not frames:
        raise ValueError("no eta images in image_files/eta")
    keep = sorted(set(range(0, len(frames), a.every)) | {len(frames) - 1})
    series, bad_color = [], 0.0
    for i in keep:
        t, path = frames[i]
        eta, off = decode(path, lut, a.size)
        bad_color = max(bad_color, off)
        m = front_metrics(eta, a.threshold)
        if m is not None:
            series.append({"frame": i, "time": t, **m, "class": classify(m, a)})
    if not series:
        raise ValueError("no deposit attached to the left boundary in any frame")
    last = series[-1]
    onset = next((s for s in series if s["class"] != "uniform"), None)
    return {
        "class": last["class"],
        "frames": len(frames),
        "final_time": last["time"],
        "onset_time": onset["time"] if onset else None,
        "front_travel": last["front_mean"] - series[0]["front_mean"],
        "short_run": (last["front_mean"] - series[0]["front_mean"]) < a.short_travel,
        "non_jet_pixels": bad_color,
        **{k: last[k] for k in ("front_mean", "front_max", "amplitude", "roughness", "length_ratio",
                                "detached_area")},
        "last_image": frames[-1][1],
        "series": series,
    }


def load_params(case_dir):
    path = os.path.join(case_dir, "params.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def plot_montage(results, lut, size, path, cols=10):
    names = sorted(results, key=lambda n: (CLASSES.index(results[n]["class"]), n))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.8 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, n in zip(axes.flat, names):
        r = results[n]
        eta, _ = decode(r["last_image"], lut, size)
        ax.imshow(eta, cmap="gray", vmin=0, vmax=1)
        for s in ax.spines.values():
            s.set_visible(True), s.set_edgecolor(COLORS[r["class"]]), s.set_linewidth(3)
        ax.axis("on"), ax.set_xticks([]), ax.set_yticks([])
        flag = " *" if r["short_run"] else ""
        ax.set_title(f"{n}{flag}\n{r['class']}  t={r['final_time']:.0f}", fontsize=7, color=COLORS[r["class"]])
    fig.suptitle("Last eta frame of each case (* = front moved less than --short-travel)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_metrics(results, a, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for n, r in results.items():
        ax.scatter(r["amplitude"], r["length_ratio"], color=COLORS[r["class"]], s=28,
                   marker="x" if r["short_run"] else "o")
        ax.annotate(n.replace("case_", ""), (r["amplitude"], r["length_ratio"]), fontsize=6,
                    xytext=(3, 2), textcoords="offset points")
    ax.axvline(a.uniform_amp, color=COLORS["uniform"], ls=":", lw=1)
    ax.axvline(a.dendrite_amp, color=COLORS["dendritic"], ls=":", lw=1)
    ax.axhline(a.uniform_len, color=COLORS["uniform"], ls=":", lw=1)
    ax.axhline(a.dendrite_len, color=COLORS["dendritic"], ls=":", lw=1)
    ax.set_xlabel("front amplitude (fraction of width)"), ax.set_ylabel("interface length / height")
    ax.set_title("last frame of each case (x = short run)")
    for c in CLASSES:
        ax.scatter([], [], color=COLORS[c], label=c)
    ax.legend(fontsize=8)

    ax = axes[1]
    for n, r in results.items():
        s = r["series"]
        ax.plot([x["time"] for x in s], [x["amplitude"] for x in s], color=COLORS[r["class"]], lw=1, alpha=0.7)
    ax.axhline(a.uniform_amp, color=COLORS["uniform"], ls=":", lw=1)
    ax.axhline(a.dendrite_amp, color=COLORS["dendritic"], ls=":", lw=1)
    ax.set_xlabel("time"), ax.set_ylabel("front amplitude"), ax.set_title("amplitude over time")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_params(results, params, path):
    names = sorted({k for p in params.values() for k in p})
    if not names:
        return False
    cols = 5
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows), squeeze=False)
    rng = np.random.default_rng(0)
    for ax in axes.flat:
        ax.axis("off")
    for ax, key in zip(axes.flat, names):
        ax.axis("on")
        for j, c in enumerate(CLASSES):
            v = [params[n][key] for n, r in results.items() if r["class"] == c and key in params[n]]
            ax.scatter(j + rng.uniform(-0.15, 0.15, len(v)), v, s=12, color=COLORS[c], alpha=0.8)
        ax.set_xticks(range(len(CLASSES))), ax.set_xticklabels(["uni", "trans", "dend"], fontsize=8)
        ax.set_title(key, fontsize=9), ax.tick_params(labelsize=7)
    fig.suptitle("Sampled parameters by morphology class", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main():
    a = parse_args()
    os.makedirs(a.output, exist_ok=True)
    lut = jet_lookup()
    case_dirs = sorted(d for d in glob.glob(os.path.join(a.data, "*")) if os.path.isdir(os.path.join(d, "image_files")))
    results, params, errors = {}, {}, {}
    for d in case_dirs:
        name = os.path.basename(d)
        try:
            results[name] = analyze_case(d, lut, a)
            params[name] = load_params(d)
            r = results[name]
            onset = f"{r['onset_time']:.1f}" if r["onset_time"] is not None else "-"
            print(f"{name}: {r['class']:<12} amp={r['amplitude']:.3f} len={r['length_ratio']:.2f} "
                  f"t_end={r['final_time']:.1f} onset={onset}{'  [short run]' if r['short_run'] else ''}")
        except Exception as exc:
            errors[name] = str(exc)
            print(f"{name}: ERROR {exc}")
    if not results:
        raise SystemExit(f"no case with image_files/eta under {a.data}")

    cols = ["case", "class", "short_run", "frames", "final_time", "onset_time", "front_mean", "front_max",
            "front_travel", "amplitude", "roughness", "length_ratio", "detached_area", "non_jet_pixels"]
    pnames = sorted({k for p in params.values() for k in p})
    with open(os.path.join(a.output, "morphology.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + pnames)
        for n, r in sorted(results.items()):
            w.writerow([n] + [r[c] for c in cols[1:]] + [params[n].get(k, "") for k in pnames])
    groups = {c: sorted(n for n, r in results.items() if r["class"] == c) for c in CLASSES}
    with open(os.path.join(a.output, "morphology.json"), "w") as f:
        json.dump({"groups": groups, "short_runs": sorted(n for n, r in results.items() if r["short_run"]),
                   "errors": errors, "settings": vars(a),
                   "cases": {n: {k: v for k, v in r.items() if k != "last_image"} for n, r in results.items()}},
                  f, indent=2)

    plot_montage(results, lut, a.size, os.path.join(a.output, "montage.png"))
    plot_metrics(results, a, os.path.join(a.output, "metrics.png"))
    plot_params(results, params, os.path.join(a.output, "params_by_class.png"))

    print()
    for c in CLASSES:
        print(f"{c} ({len(groups[c])}): {' '.join(groups[c]) or '-'}")
    short = [n for n in groups["uniform"] if results[n]["short_run"]]
    if short:
        print(f"uniform but the front moved less than {a.short_travel:.0%} of the width "
              f"(the run may be too short): {' '.join(short)}")
    if errors:
        print(f"errors ({len(errors)}): {' '.join(sorted(errors))}")
    print(f"results in {a.output}")


if __name__ == "__main__":
    main()
