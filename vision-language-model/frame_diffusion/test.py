#!/usr/bin/env python3
"""Roll out a trained frame-diffusion model and compare it with simulation frames.

python frame_diffusion/test.py --run runs/fd_k4
python frame_diffusion/test.py --run runs/fd_k4 --case case_001 --start 10 --num-frames 20 --samples 4
python frame_diffusion/test.py --run runs/fd_k4 --case case_066 --t0 150 --num-frames 10 --param POT_LEFT=-0.4
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
from diffusers import DDIMScheduler
from scipy import ndimage
from skimage.metrics import structural_similarity

from data import CHANNELS, Normalizer, ParamNormalizer, discover_cases, load_case, load_params, log_gaps, save_json
from model import FramePredictor, sample_next


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="training output directory")
    p.add_argument("--data", help="data root (default: the one used in training)")
    p.add_argument("--case", help="case to test (default: every case with a test range)")
    p.add_argument("--start", type=int, help="index of the first predicted frame (default: test split start)")
    p.add_argument("--t0", type=float,
                   help="predict the frames after this time; the context is the K frames at or before t0")
    p.add_argument("--params-file", help="JSON with all simulation parameters (default: the case's own file)")
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                   help="override one parameter, e.g. --param POT_LEFT=-0.4 (repeatable)")
    p.add_argument("--num-frames", type=int, help="frames to predict (default: to the end of the data)")
    p.add_argument("--dt", type=float, help="time step for frames beyond the available data")
    p.add_argument("--teacher-forcing", action="store_true",
                   help="condition every step on ground-truth frames (one-step accuracy)")
    p.add_argument("--samples", type=int, default=1, help="stochastic rollouts per case (ensemble)")
    p.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0 = deterministic given noise)")
    p.add_argument("--threshold", type=float, default=0.5, help="eta threshold for the deposit phase")
    p.add_argument("--output", help="default: <run>/test")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def front_metrics(eta, threshold):
    """Tip x (fraction of domain width) and area fraction of deposit attached to the left boundary."""
    labels, _ = ndimage.label(eta >= threshold)  # 4-connectivity
    attached = np.isin(labels, np.setdiff1d(np.unique(labels[:, 0]), [0]))
    cols = np.where(attached.any(axis=0))[0]
    return {"tip_x": float(cols.max() / (eta.shape[1] - 1)) if cols.size else 0.0, "area_fraction": float(attached.mean())}


def field_metrics(pred, gt, lo, hi, threshold):
    out = {}
    for c, name in enumerate(CHANNELS):
        rng = float(hi[c] - lo[c]) or 1.0
        mse = float(np.mean((pred[c] - gt[c]) ** 2))
        out[name] = {
            "mse": mse,
            "psnr": float(10 * np.log10(rng ** 2 / max(mse, 1e-20))),
            "ssim": float(structural_similarity(gt[c], pred[c], data_range=rng)),
        }
    p, g = front_metrics(pred[0], threshold), front_metrics(gt[0], threshold)
    out["front"] = {"pred": p, "gt": g, "tip_error": p["tip_x"] - g["tip_x"]}
    return out


def plot_comparison(times, gt, preds, path, max_cols=6):
    idx = np.unique(np.linspace(0, len(times) - 1, min(max_cols, len(times))).astype(int))
    rows = [(f"{n} {kind}", c, kind) for c, n in enumerate(CHANNELS) for kind in ("GT", "pred", "|err|")]
    fig, axes = plt.subplots(len(rows), len(idx), figsize=(2.2 * len(idx), 2.1 * len(rows)), squeeze=False)
    for r, (label, c, kind) in enumerate(rows):
        vmin, vmax = gt[:, c].min(), gt[:, c].max()
        for j, k in enumerate(idx):
            ax = axes[r, j]
            if kind == "GT":
                ax.imshow(gt[k, c], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            elif kind == "pred":
                ax.imshow(preds[k, c], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            else:
                ax.imshow(np.abs(preds[k, c] - gt[k, c]), origin="lower", cmap="magma", vmin=0,
                          vmax=(vmax - vmin) * 0.5)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={times[k]:.1f}", fontsize=8)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves(times, metrics, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for name in CHANNELS:
        axes[0].plot(times, [m[name]["psnr"] for m in metrics], label=name)
        axes[1].plot(times, [m[name]["ssim"] for m in metrics], label=name)
    axes[2].plot(times, [m["front"]["gt"]["tip_x"] for m in metrics], "k-", label="GT tip x")
    axes[2].plot(times, [m["front"]["pred"]["tip_x"] for m in metrics], "r--", label="pred tip x")
    for ax, title in zip(axes, ("PSNR (dB)", "SSIM", "Dendrite tip position")):
        ax.set_title(title), ax.set_xlabel("time"), ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_rollout_gif(times, gt, preds, path, fps=4):
    """Animation with one column per channel: GT on top, prediction below."""
    fig, axes = plt.subplots(2, len(CHANNELS), figsize=(2.6 * len(CHANNELS), 5.4), squeeze=False)
    ims = []
    for c, name in enumerate(CHANNELS):
        lo, hi = gt[:, c].min(), gt[:, c].max()
        for r, (arr, kind) in enumerate(((gt, "GT"), (preds, "pred"))):
            ax = axes[r, c]
            ims.append(ax.imshow(arr[0, c], origin="lower", cmap="viridis", vmin=lo, vmax=hi))
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{name} {kind}", fontsize=9)
    title = fig.suptitle("")
    fig.tight_layout()

    def draw(k):
        for c in range(len(CHANNELS)):
            ims[2 * c].set_data(gt[k, c])
            ims[2 * c + 1].set_data(preds[k, c])
        title.set_text(f"frame {k + 1}/{len(times)}   t = {times[k]:.2f}")
        return ims

    FuncAnimation(fig, draw, frames=len(times)).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


@torch.no_grad()
def rollout(model, scheduler, normalizer, frames, times, start, n, args, generator, device, params=None):
    """Predict frames start..start+n-1; frames beyond the data use args.dt spacing."""
    K = model.config.num_context
    hist = [normalizer.encode(frames[i]).to(device) for i in range(start - K, start)]
    hist_t = list(times[start - K:start])
    preds, pred_t = [], []
    for k in range(n):
        i = start + k
        t_next = times[i] if i < len(times) else hist_t[-1] + args.dt
        gaps = log_gaps(np.array(hist_t[-K:] + [t_next]))[None].to(device)
        ctx = torch.cat(hist[-K:])[None]
        x = sample_next(model, scheduler, ctx, gaps, args.steps, args.eta, generator, params=params)[0]
        preds.append(normalizer.decode(x).float().cpu())
        pred_t.append(t_next)
        if args.teacher_forcing and i < len(times):
            hist.append(normalizer.encode(frames[i]).to(device))
        else:
            hist.append(x)
        hist_t.append(t_next)
    return torch.stack(preds).numpy(), np.array(pred_t)


def main():
    args = parse_args()
    with open(os.path.join(args.run, "run_config.json")) as f:
        cfg = json.load(f)
    train_args = cfg["args"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FramePredictor.from_pretrained(os.path.join(args.run, "model")).to(device).eval()
    scheduler = DDIMScheduler.from_pretrained(os.path.join(args.run, "scheduler"))
    normalizer = Normalizer.from_dict(cfg["normalizer"])
    lo, hi = np.array(cfg["normalizer"]["lo"]), np.array(cfg["normalizer"]["hi"])
    K = model.config.num_context
    out_root = args.output or os.path.join(args.run, "test" + ("_tf" if args.teacher_forcing else ""))
    param_norm = ParamNormalizer.from_dict(cfg["param_normalizer"]) if cfg.get("param_normalizer") else None
    overrides = {}
    for item in args.param:
        key, _, value = item.partition("=")
        if not param_norm or key not in param_norm.names:
            raise ValueError(f"unknown parameter {key!r}; the model uses {param_norm.names if param_norm else 'none'}")
        overrides[key] = float(value)

    data_root = args.data or train_args["data"]
    names = [args.case] if args.case else list(cfg["test_ranges"])
    if not names:
        raise ValueError("no test range in run_config.json; pass --case and --start")
    case_dirs = discover_cases(data_root, names)

    summary = {}
    for name, case_dir in case_dirs.items():
        frames, times = load_case(case_dir, train_args["resolution"])
        if args.t0 is not None:
            start = int(np.searchsorted(times, args.t0, side="right"))
        else:
            start = args.start if args.start is not None else cfg["test_ranges"].get(name, [K])[0]
        if start < K:
            raise ValueError(f"need {K} context frames before the first prediction; "
                             f"t0/--start leaves only {start} (first times: {np.round(times[:K + 1], 3).tolist()})")

        params, used = None, None
        if param_norm:
            if args.params_file:
                with open(args.params_file) as f:
                    used = json.load(f)
            else:
                keys, values = load_params({name: case_dir})
                used = dict(zip(keys, values[name].tolist()))
            used.update(overrides)
            missing = [k for k in param_norm.names if k not in used]
            if missing:
                raise ValueError(f"parameters missing for {name}: {missing}")
            params = param_norm.encode([used[k] for k in param_norm.names])[None].to(device)
        n = args.num_frames or (len(times) - start)
        if start + n > len(times) and args.dt is None:
            raise ValueError("predicting past the last frame needs --dt")
        n_gt = max(0, min(n, len(times) - start))
        out_dir = os.path.join(out_root, name)
        os.makedirs(out_dir, exist_ok=True)

        runs = []
        for s in range(args.samples):
            g = torch.Generator(device=device).manual_seed(args.seed + s)
            preds, pred_t = rollout(model, scheduler, normalizer, frames, times, start, n, args, g, device, params)
            runs.append(preds)
            sample_dir = os.path.join(out_dir, f"sample_{s:02d}", "npy_files")
            os.makedirs(sample_dir, exist_ok=True)
            for p, t in zip(preds, pred_t):  # same (H, W, C) layout as step0
                np.save(os.path.join(sample_dir, f"{t:.6f}.npy"), p.transpose(1, 2, 0))
        runs = np.stack(runs)  # (S, n, C, H, W)
        mean = runs.mean(0)

        info = {"start": start, "context_times": times[start - K:start].tolist(), "num_frames": n,
                "params": used, "params_changed": sorted(overrides) or bool(args.params_file)}
        if n_gt == 0:
            summary[name] = {**info, "note": "no ground truth in range"}
            continue
        gt = frames[start:start + n_gt].numpy()
        per_frame = [field_metrics(mean[k], gt[k], lo, hi, args.threshold) for k in range(n_gt)]
        for k, m in enumerate(per_frame):
            m["time"] = float(pred_t[k])
            m["ensemble_std"] = runs[:, k].std(0).mean(axis=(1, 2)).tolist() if args.samples > 1 else None
        save_json(per_frame, os.path.join(out_dir, "metrics.json"))
        plot_comparison(pred_t[:n_gt], gt, mean[:n_gt], os.path.join(out_dir, "comparison.png"))
        plot_curves(pred_t[:n_gt], per_frame, os.path.join(out_dir, "metrics.png"))
        plot_rollout_gif(pred_t[:n_gt], gt, mean[:n_gt], os.path.join(out_dir, "rollout.gif"))
        summary[name] = {
            **info, "frames_with_gt": n_gt, "samples": args.samples,
            "mode": "teacher_forcing" if args.teacher_forcing else "autoregressive",
            **{c: {k: float(np.mean([m[c][k] for m in per_frame])) for k in ("mse", "psnr", "ssim")}
               for c in CHANNELS},
            "final_tip_error": per_frame[-1]["front"]["tip_error"],
        }
        print(f"{name}: " + ", ".join(f"{c} PSNR {summary[name][c]['psnr']:.2f} SSIM {summary[name][c]['ssim']:.3f}"
                                      for c in CHANNELS))
    save_json(summary, os.path.join(out_root, "summary.json"))
    print(f"results in {out_root}")


if __name__ == "__main__":
    main()
