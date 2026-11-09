#!/usr/bin/env python3
"""Check whether a trained frame-diffusion model reproduces the MOOSE simulations, with Qwen reading the results.

The simulations contain Langevin noise, so the surrogate is judged statistically: from the same start
frame and parameters, S rollouts are drawn and their morphology metrics (step1's front metrics and
classes, plus Li inventory and eta bounds) are compared with the MOOSE frames.

Numerical checks (pass/fail decided by code, thresholds below):
  class_agreement     majority class of the samples at the last frame equals the MOOSE class
  seed_ratio_<m>      surrogate error / error expected from the noise alone. The noise level comes from
                      MOOSE cases that share all parameters except the seed (--seed-keys). For a perfect
                      surrogate the ratio is about 1. Not evaluated without such replicate cases.
  coverage_<m>        fraction of frames whose MOOSE value lies in the central --interval of the samples
  eta_oob, li_rel_err physical checks: eta outside [-0.05, 1.05]; Li inventory mean(c + li_ratio*eta)
  sweep_direction     optional (--sweep): the surrogate's trend of a final metric along one parameter has
                      the sign of the MOOSE trend across cases (Spearman; MOOSE trend is confounded by the
                      other parameters, so it is a reference only)
The trusted horizon is the number of rollout frames before the median front error exceeds --horizon-tol.

Qwen (Ollama) then reads the results:
  compare   per case, MOOSE and surrogate samples side by side; the answers (morphology, faster/slower,
            rougher/smoother, detached fragments) are checked against the metrics, which measures how far
            Qwen's reading can be trusted
  blind     per case, one MOOSE row and one surrogate row in random order; Qwen picks the simulation.
            Accuracy near 50% means Qwen cannot tell them apart (a perceptual check, not a proof)
  verdict   Qwen reviews the numbers and figures and writes the conclusion; pass/fail stays with the code,
            and cited evidence keys and the overall rating are checked for consistency

Run from vision-language-model/:
python frame_diffusion/validate_surrogate.py --run runs/<run>
python frame_diffusion/validate_surrogate.py --run runs/<run> --data /path/to/data --samples 8 --sweep POT_LEFT
python frame_diffusion/validate_surrogate.py --run runs/<run> --no-qwen
python frame_diffusion/validate_surrogate.py --run runs/<run> --qwen-only   # redo the Qwen part on saved results
"""

from __future__ import annotations

import argparse
import base64
import collections
import glob
import importlib.util
import io
import json
import math
import os
import random
import sys
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import List, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))  # step1_classify_morphology.py is in vision-language-model/
import step1_classify_morphology as step1  # noqa: E402

from data import (Normalizer, ParamNormalizer, discover_cases, frames_to_tensor, load_case, load_params,  # noqa: E402
                  save_json)
from model import FramePredictor  # noqa: E402

# test.py is loaded by path: "import test" could pick up Python's own test package.
_spec = importlib.util.spec_from_file_location("fd_test", os.path.join(HERE, "test.py"))
fd_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fd_test)

# step1's default class thresholds, so the classes match data/summary/morphology.
STEP1_CLASS = SimpleNamespace(uniform_amp=0.03, uniform_len=1.10, dendrite_amp=0.10, dendrite_len=1.50)
CLASSES = ("uniform", "transitional", "dendritic", "none")  # none: no deposit attached to the left edge
METRICS = ("front_mean", "front_max", "amplitude", "roughness", "length_ratio", "detached_area")
KEY_METRICS = ("front_mean", "amplitude")
SWEEP_METRICS = ("front_mean", "amplitude", "length_ratio")
MORPH_TO_CLASS = {"uniform": "uniform", "wavy": "transitional", "dendritic": "dendritic"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="training output directory")
    p.add_argument("--data", help="data root (default: the one used in training)")
    p.add_argument("--cases", nargs="*", help="cases to test (default: the run's test cases)")
    p.add_argument("--from-start", action="store_true",
                   help="roll out from the first frames instead of the test split start "
                        "(with a time split this includes training frames)")
    p.add_argument("--num-frames", type=int, help="limit the rollout length")
    p.add_argument("--samples", type=int, default=8, help="rollouts per case")
    p.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    p.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance on the parameters")
    p.add_argument("--threshold", type=float, default=0.5, help="eta threshold for the deposit")
    p.add_argument("--li-ratio", type=float, default=76.9, help="c_s/c_0 of the template (Li inventory)")
    p.add_argument("--interval", type=float, default=0.9, help="central sample interval for coverage")
    p.add_argument("--seed-keys", nargs="*", default=["SEED", "seed"],
                   help="params.json keys ignored when grouping replicate cases")
    p.add_argument("--max-seed-groups", type=int, default=30)
    p.add_argument("--sweep", nargs="*", default=[], metavar="PARAM", help="parameters for the direction check")
    p.add_argument("--sweep-points", type=int, default=7)
    p.add_argument("--sweep-cases", type=int, default=3)
    p.add_argument("--sweep-samples", type=int, default=4)
    p.add_argument("--rho-min", type=float, default=0.3, help="|Spearman| below this is inconclusive")
    p.add_argument("--pass-class", type=float, default=0.9)
    p.add_argument("--pass-ratio", type=float, default=1.5)
    p.add_argument("--pass-coverage", type=float, default=0.8)
    p.add_argument("--pass-oob", type=float, default=0.01)
    p.add_argument("--pass-li", type=float, default=0.05)
    p.add_argument("--horizon-tol", type=float, default=0.05, help="front error (fraction of width)")
    p.add_argument("--front-tol", type=float, default=0.03,
                   help="final front difference counted as ahead/behind when checking Qwen")
    p.add_argument("--amp-tol", type=float, default=0.02,
                   help="final amplitude difference counted as rougher/smoother when checking Qwen")
    p.add_argument("--no-qwen", action="store_true")
    p.add_argument("--qwen-only", action="store_true", help="skip the rollouts and reuse saved results")
    p.add_argument("--qwen-model", default="qwen3-vl:8b-instruct")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--qwen-rows", type=int, default=2, help="surrogate samples shown to Qwen")
    p.add_argument("--panel-frames", type=int, default=5, help="times shown to Qwen")
    p.add_argument("--lang", default="Chinese", help="language of Qwen's text fields")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--output", help="default: <run>/validation")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# ---------------------------------------------------------------- metrics

def frame_metrics(frame, a):
    """Metrics of one (C, H, W) physical frame; eta rows are y, the deposit grows from column 0."""
    eta = frame[0]
    m = step1.front_metrics(eta, a.threshold)
    if m is None:
        m, cls = {k: 0.0 for k in METRICS}, "none"
    else:
        cls = step1.classify(m, STEP1_CLASS)
    m = {k: m[k] for k in METRICS}
    m["li"] = float(np.mean(frame[1] + a.li_ratio * eta))
    m["eta_oob"] = float(np.mean((eta < -0.05) | (eta > 1.05)))
    m["class"] = cls
    return m


def series(frames, a):
    """(T, C, H, W) -> {metric: (T,) array, 'class': [T]}."""
    rows = [frame_metrics(f, a) for f in frames]
    out = {k: np.array([r[k] for r in rows]) for k in (*METRICS, "li", "eta_oob")}
    out["class"] = [r["class"] for r in rows]
    return out


def onset(classes, times):
    return next((float(t) for c, t in zip(classes, times) if c in ("transitional", "dendritic")), None)


def majority(classes):
    return collections.Counter(classes).most_common(1)[0][0]


def tolist(x):
    if isinstance(x, dict):
        return {k: tolist(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tolist(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# ---------------------------------------------------------------- surrogate

def load_run(a):
    with open(os.path.join(a.run, "run_config.json")) as f:
        cfg = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FramePredictor.from_pretrained(os.path.join(a.run, "model")).to(device).eval()
    if a.guidance != 1.0 and getattr(model, "null_param", None) is None:
        raise ValueError("--guidance needs a model trained with --param-dropout > 0")
    targs = cfg["args"]
    return SimpleNamespace(
        cfg=cfg, device=device, model=model, K=model.config.num_context,
        scheduler=DDIMScheduler.from_pretrained(os.path.join(a.run, "scheduler")),
        normalizer=Normalizer.from_dict(cfg["normalizer"]),
        param_norm=ParamNormalizer.from_dict(cfg["param_normalizer"]) if cfg.get("param_normalizer") else None,
        res=targs["resolution"], dt=targs.get("dt"), data=a.data or targs["data"],
        roll=SimpleNamespace(steps=a.steps, eta=0.0, dt=targs.get("dt"), teacher_forcing=False, guidance=a.guidance),
    )


def encode_params(ctx, values):
    if not ctx.param_norm:
        return None
    return ctx.param_norm.encode([values[k] for k in ctx.param_norm.names])[None].to(ctx.device)


def case_params(ctx, name, case_dir):
    if not ctx.param_norm:
        return None
    keys, values = load_params({name: case_dir}, ctx.param_norm.names)
    return dict(zip(keys, values[name].tolist()))


def sample_rollouts(ctx, frames, times, start, n, params, samples, seed):
    runs = []
    for s in range(samples):
        g = torch.Generator(device=ctx.device).manual_seed(seed + s)
        preds, _ = fd_test.rollout(ctx.model, ctx.scheduler, ctx.normalizer, frames, times, start, n, ctx.roll,
                                   g, ctx.device, params)
        runs.append(preds)
    return np.stack(runs)  # (S, n, C, H, W)


def evaluate_case(name, case_dir, ctx, a, out_dir):
    frames, times = load_case(case_dir, ctx.res, ctx.dt)
    test_ranges = ctx.cfg.get("test_ranges", {})
    start = ctx.K if (a.from_start or name not in test_ranges) else test_ranges[name][0]
    n = len(times) - start
    if a.num_frames:
        n = min(n, a.num_frames)
    if n <= 0:
        print(f"[WARN] {name}: no frames after the start ({start}); skipped")
        return None
    used = case_params(ctx, name, case_dir)
    runs = sample_rollouts(ctx, frames, times, start, n, encode_params(ctx, used), a.samples, a.seed)
    gt = frames[start:start + n].numpy()
    t = times[start:start + n]
    g = series(gt, a)
    preds = [series(r, a) for r in runs]

    lo_q, hi_q = (1 - a.interval) / 2, (1 + a.interval) / 2
    per_metric = {}
    for k in METRICS:
        P = np.stack([p[k] for p in preds])  # (S, n)
        mean = P.mean(0)
        lo, hi = np.quantile(P, [lo_q, hi_q], axis=0)
        per_metric[k] = {"gt": g[k], "mean": mean, "std": P.std(0), "lo": lo, "hi": hi,
                         "abs_err": np.abs(mean - g[k]), "covered": (g[k] >= lo) & (g[k] <= hi)}
    li = np.stack([p["li"] for p in preds])
    li_rel = np.median(np.abs(li - g["li"]) / np.maximum(np.abs(g["li"]), 1e-12), axis=0)
    final_classes = [p["class"][-1] for p in preds]
    pred_onsets = [onset(p["class"], t) for p in preds]
    gt_onset = onset(g["class"], t)
    both = [o for o in pred_onsets if o is not None]

    result = {
        "case": name, "start": start, "num_frames": n, "times": t, "params": used,
        "leakage": name in ctx.cfg.get("train_ranges", {}) and (a.from_start or name not in test_ranges),
        "metrics": per_metric, "li_rel_err": li_rel,
        "eta_oob": float(np.mean([p["eta_oob"].mean() for p in preds])),
        "gt_class": g["class"][-1], "sample_classes": final_classes, "majority_class": majority(final_classes),
        "class_agree": majority(final_classes) == g["class"][-1],
        "sample_class_agree": float(np.mean([c == g["class"][-1] for c in final_classes])),
        "gt_onset": gt_onset, "sample_onsets": pred_onsets,
        "onset_abs_err": float(np.mean([abs(o - gt_onset) for o in both])) if both and gt_onset is not None else None,
        "final_front_diff": float(per_metric["front_mean"]["mean"][-1] - g["front_mean"][-1]),
        "final_amp_diff": float(per_metric["amplitude"]["mean"][-1] - g["amplitude"][-1]),
        "max_sample_detached": float(max(p["detached_area"][-1] for p in preds)),
        "gt_detached": float(g["detached_area"][-1]),
    }
    plot_case(result, os.path.join(out_dir, f"{name}_curves.png"))

    # Frames for Qwen: eta only, evenly spaced over the rollout.
    idx = np.unique(np.linspace(0, n - 1, min(a.panel_frames, n)).astype(int))
    np.savez_compressed(os.path.join(out_dir, f"{name}_panel.npz"), times=t[idx], gt=gt[idx, 0],
                        samples=runs[:max(a.qwen_rows, 1), idx, 0])
    save_json(tolist(result), os.path.join(out_dir, f"{name}.json"))
    print(f"{name}: class {g['class'][-1]} vs samples {final_classes}; "
          f"final front diff {result['final_front_diff']:+.3f}")
    return result


# ---------------------------------------------------------------- noise floor and sweep

def seed_floor(ctx, a):
    """Noise level per metric from MOOSE cases that differ only in the seed: sigma ~ median|a-b|/sqrt(2)."""
    try:
        dirs = discover_cases(ctx.data)
        names, raw = load_params(dirs)
    except (FileNotFoundError, ValueError) as e:
        return {"groups": 0, "note": f"not evaluated: {e}"}
    keep = [i for i, k in enumerate(names) if k not in set(a.seed_keys)]
    groups = collections.defaultdict(list)
    for n, v in raw.items():
        groups[tuple(float(f"{v[i]:.6g}") for i in keep)].append(n)
    groups = [g for g in groups.values() if len(g) >= 2][:a.max_seed_groups]
    if not groups:
        return {"groups": 0, "note": "no MOOSE cases share all parameters except the seed"}
    diffs = collections.defaultdict(list)
    for g in groups:
        curves = {}
        for n in g:
            frames, times = load_case(dirs[n], ctx.res, ctx.dt)
            curves[n] = (times, series(frames.numpy(), a))
        for x, y in ((x, y) for i, x in enumerate(g) for y in g[i + 1:]):
            (tx, sx), (ty, sy) = curves[x], curves[y]
            common = tx[(tx >= ty[0]) & (tx <= ty[-1])]
            for k in METRICS:
                diffs[k].extend(np.abs(np.interp(common, tx, sx[k]) - np.interp(common, ty, sy[k])).tolist())
    sigma = {k: float(np.median(v) / math.sqrt(2)) for k, v in diffs.items() if v}
    return {"groups": len(groups), "pairs": sum(len(g) * (len(g) - 1) // 2 for g in groups), "sigma": sigma}


def final_metric_at(case_dir, t_ref, a, res):
    """step1 metrics of the saved frame closest to t_ref, or None if the case ends before t_ref."""
    files = glob.glob(os.path.join(case_dir, "npy_files", "*.npy"))
    times = np.array([float(os.path.basename(f)[:-4]) for f in files])
    if not len(times) or times.max() < t_ref:
        return None
    f = files[int(np.argmin(np.abs(times - t_ref)))]
    return frame_metrics(frames_to_tensor([np.load(f)], res)[0].numpy(), a)


def sweep(param, results, ctx, a, out_dir):
    if not ctx.param_norm or param not in ctx.param_norm.names:
        raise ValueError(f"--sweep {param}: the model is conditioned on {ctx.param_norm.names if ctx.param_norm else 'nothing'}")
    dirs = discover_cases(ctx.data)
    train = [n for n in ctx.cfg.get("train_ranges", {}) if n in dirs]
    _, raw = load_params({n: dirs[n] for n in train}, ctx.param_norm.names)
    vals = np.array([raw[n][ctx.param_norm.names.index(param)] for n in train])
    lo, hi = vals.min(), vals.max()
    values = np.geomspace(lo, hi, a.sweep_points) if param in ctx.param_norm.log and lo > 0 \
        else np.linspace(lo, hi, a.sweep_points)

    curves = []
    for r in results[:a.sweep_cases]:
        frames, times = load_case(dirs[r["case"]], ctx.res, ctx.dt)
        n = min(len(times) - ctx.K, a.num_frames or len(times))
        rows = []
        for v in values:
            runs = sample_rollouts(ctx, frames, times, ctx.K, n, encode_params(ctx, {**r["params"], param: float(v)}),
                                   a.sweep_samples, a.seed)
            last = [frame_metrics(run[-1], a) for run in runs]
            rows.append({k: float(np.mean([m[k] for m in last])) for k in SWEEP_METRICS})
        curves.append({"case": r["case"], "t_ref": float(times[ctx.K + n - 1]),
                       "curve": {k: [row[k] for row in rows] for k in SWEEP_METRICS}})

    # MOOSE reference: rank correlation across all cases at the sweep's final time (confounded).
    t_ref = min(c["t_ref"] for c in curves)
    ref_p, ref_m = [], collections.defaultdict(list)
    _, all_raw = load_params(dirs, ctx.param_norm.names)
    for n in dirs:
        m = final_metric_at(dirs[n], t_ref, a, ctx.res)
        if m is not None:
            ref_p.append(all_raw[n][ctx.param_norm.names.index(param)])
            for k in SWEEP_METRICS:
                ref_m[k].append(m[k])

    def rho(x, y):
        return float(spearmanr(x, y).correlation) if len(x) >= 5 and np.std(x) > 0 and np.std(y) > 0 else None

    checks = {}
    for k in SWEEP_METRICS:
        moose = rho(ref_p, ref_m[k])
        for c in curves:
            sur = rho(values, c["curve"][k]) if len(values) >= 5 else None
            if moose is None or sur is None or abs(moose) < a.rho_min or abs(sur) < a.rho_min:
                agree = None
            else:
                agree = bool(np.sign(moose) == np.sign(sur))
            checks[f"{c['case']}:{k}"] = {"surrogate_rho": sur, "moose_rho": moose, "agree": agree}
    out = {"param": param, "values": values, "t_ref": t_ref, "moose_cases": len(ref_p),
           "curves": curves, "checks": checks}
    save_json(tolist(out), os.path.join(out_dir, f"sweep_{param}.json"))
    fig, axes = plt.subplots(1, len(SWEEP_METRICS), figsize=(4.2 * len(SWEEP_METRICS), 3.4))
    for ax, k in zip(axes, SWEEP_METRICS):
        for c in curves:
            ax.plot(values, c["curve"][k], "o-", label=c["case"])
        ax.set_xlabel(param), ax.set_title(f"{k} at t={t_ref:.1f}")
        if param in ctx.param_norm.log:
            ax.set_xscale("log")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"sweep_{param}.png"), dpi=120)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- summary

def summarize(results, floor, sweeps, a, ctx):
    S = a.samples
    crit = {}

    def add(name, value, threshold, ok):
        crit[name] = {"value": value, "threshold": threshold, "passed": None if value is None else bool(ok)}

    agree = float(np.mean([r["class_agree"] for r in results]))
    add("class_agreement", agree, f">= {a.pass_class}", agree >= a.pass_class)
    for k in KEY_METRICS:
        err = float(np.median(np.concatenate([r["metrics"][k]["abs_err"] for r in results])))
        sigma = (floor.get("sigma") or {}).get(k)
        ratio = err / (sigma * math.sqrt(1 + 1 / S)) if sigma else None
        add(f"seed_ratio_{k}", ratio, f"<= {a.pass_ratio}", ratio is not None and ratio <= a.pass_ratio)
        cov = float(np.mean(np.concatenate([r["metrics"][k]["covered"] for r in results])))
        add(f"coverage_{k}", cov, f">= {a.pass_coverage} (interval {a.interval}, {S} samples)", cov >= a.pass_coverage)
    oob = float(max(r["eta_oob"] for r in results))
    add("eta_oob", oob, f"<= {a.pass_oob}", oob <= a.pass_oob)
    li = float(np.median(np.concatenate([r["li_rel_err"] for r in results])))
    add("li_rel_err", li, f"<= {a.pass_li}", li <= a.pass_li)
    votes = [c["agree"] for s in sweeps for c in s["checks"].values() if c["agree"] is not None]
    add("sweep_direction", float(np.mean(votes)) if votes else None, "== 1.0", votes and all(votes))

    # Error against rollout step, pooled over cases.
    L = max(r["num_frames"] for r in results)
    horizon_err = {}
    for k in KEY_METRICS:
        horizon_err[k] = [float(np.median([r["metrics"][k]["abs_err"][i] for r in results if r["num_frames"] > i]))
                          for i in range(L)]
    bad = [i for i, e in enumerate(horizon_err["front_mean"]) if e > a.horizon_tol]
    horizon = bad[0] if bad else L
    confusion = {g: {p: sum(r["gt_class"] == g and r["majority_class"] == p for r in results) for p in CLASSES}
                 for g in CLASSES}
    onset_err = [r["onset_abs_err"] for r in results if r["onset_abs_err"] is not None]
    return {
        "run": a.run, "cases": [r["case"] for r in results], "samples": S, "guidance": a.guidance,
        "dt": ctx.dt, "leakage_cases": [r["case"] for r in results if r["leakage"]],
        "criteria": crit, "trusted_horizon_frames": horizon,
        "trusted_horizon_time": horizon * ctx.dt if ctx.dt else None,
        "horizon_error": horizon_err, "confusion": confusion,
        "onset_abs_err_mean": float(np.mean(onset_err)) if onset_err else None,
        "seed_floor": floor, "sweeps": [{"param": s["param"], "checks": s["checks"]} for s in sweeps],
        "per_case": {r["case"]: {"gt_class": r["gt_class"], "sample_classes": r["sample_classes"],
                                 "final_front_diff": r["final_front_diff"], "final_amp_diff": r["final_amp_diff"],
                                 "onset_abs_err": r["onset_abs_err"]} for r in results},
    }


def plot_case(r, path):
    keys = ("front_mean", "amplitude", "length_ratio")
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(4.0 * (len(keys) + 1), 3.2))
    t = r["times"]
    for ax, k in zip(axes, keys):
        m = r["metrics"][k]
        ax.fill_between(t, m["lo"], m["hi"], color="tab:red", alpha=0.2, label="surrogate interval")
        ax.plot(t, m["mean"], "r--", label="surrogate mean")
        ax.plot(t, m["gt"], "k-", label="MOOSE")
        ax.set_title(k), ax.set_xlabel("time")
    axes[-1].plot(t, r["li_rel_err"], "b-")
    axes[-1].set_title("Li inventory rel. error"), axes[-1].set_xlabel("time")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{r['case']}: MOOSE {r['gt_class']}, samples {', '.join(r['sample_classes'])}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_summary(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for k, e in summary["horizon_error"].items():
        axes[0].plot(np.arange(1, len(e) + 1), e, label=k)
    axes[0].axvline(summary["trusted_horizon_frames"] + 0.5, color="k", ls=":", label="trusted horizon")
    axes[0].set_xlabel("rollout frame"), axes[0].set_title("median |surrogate mean - MOOSE|"), axes[0].legend(fontsize=8)
    crit = summary["criteria"]
    names = [n for n in crit if crit[n]["value"] is not None]
    colors = ["tab:green" if crit[n]["passed"] else "tab:red" for n in names]
    axes[1].barh(names, [crit[n]["value"] for n in names], color=colors)
    axes[1].set_title("criteria (green = passed)"), axes[1].tick_params(axis="y", labelsize=8)
    M = np.array([[summary["confusion"][g][p] for p in CLASSES] for g in CLASSES])
    axes[2].imshow(M, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            axes[2].text(j, i, str(M[i, j]), ha="center", va="center")
    axes[2].set_xticks(range(len(CLASSES))), axes[2].set_xticklabels(CLASSES, fontsize=8)
    axes[2].set_yticks(range(len(CLASSES))), axes[2].set_yticklabels(CLASSES, fontsize=8)
    axes[2].set_xlabel("surrogate (majority)"), axes[2].set_ylabel("MOOSE"), axes[2].set_title("final-frame class")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- Qwen

CONVENTIONS = """Each panel shows the order parameter eta of a phase-field model of lithium electrodeposition,
with the same fixed color scale [0, 1] everywhere: yellow = lithium metal (eta = 1), dark purple = electrolyte (eta = 0).
x points right and y up; the lithium deposit grows from the left edge toward the right. Columns are times
t = {times} (simulation units)."""

COMPARE_PROMPT = """You are checking a neural-network surrogate of a numerical simulation.
""" + CONVENTIONS + """
Row "MOOSE" is the reference numerical simulation. Rows {rows} are independent samples of the surrogate, started from the
same frame with the same parameters. The simulation contains random noise, so the surrogate should match the reference
statistically (growth speed, front shape, amount of branching), not pixel by pixel.
Answer only from the image, in the JSON format given. Write the text fields in {lang}.
1. morphology_moose: the last MOOSE frame is uniform (flat front), wavy (undulating, no fingers) or dendritic (clear fingers or branches).
2. morphology_surrogate: the same for the last frame of each surrogate row, in row order ({nrows} values).
3. same_morphology: whether the surrogate rows show the same kind of morphology as MOOSE.
4. differences: systematic differences of the surrogate from MOOSE at the last time, chosen from the allowed values
   (front_ahead / front_behind: the deposit reaches further / less far to the right; rougher / smoother: the front varies
   more / less across y); empty list if there are none.
5. artifacts: visual problems in the surrogate rows that a physical simulation would not show (blur, checkerboard,
   speckle noise, floating fragments, stripes); empty list if none.
6. similarity: 1 (unrelated) to 5 (the same kind of evolution).
7. evidence: one sentence per column about what you see, naming the time.
8. summary: two or three sentences. Describe only what is visible; do not estimate numbers.
"""

BLIND_PROMPT = CONVENTIONS + """
The image has two rows, A and B, showing the same times. One row comes from a numerical (finite-element) simulation, the
other was generated by a neural network that imitates it. Decide which row is the numerical simulation.
Answer in the JSON format given; write the reason in {lang}.
"""

VERDICT_PROMPT = """You review the validation of a diffusion-model surrogate of a phase-field simulation of lithium
dendrite growth. The JSON below holds the results computed by code; the figure shows (left) the error against rollout
length, (middle) the criteria, green = passed, and (right) the final-frame morphology classes.
Rules:
- The pass/fail of every criterion was computed by code and is final. A criterion with passed = null was not evaluated.
- overall: "reliable" only if every evaluated criterion passed; "not_reliable" if class_agreement or a seed_ratio failed;
  otherwise "reliable_with_limits".
- Every finding must cite evidence keys, written as paths into the JSON with dots, e.g. "criteria.class_agreement" or
  "per_case.case_005.final_front_diff". Write only the key, without its value. Use only keys that exist.
- qwen_reading holds how often your own reading of the images agreed with the metrics, and the blind test accuracy;
  take it into account when weighing visual statements.
- failure_modes: where and how the surrogate fails (cases, classes, rollout length). recommendations: concrete next steps.
Write the text fields in {lang}.

{evidence}
"""

Morph = Literal["uniform", "wavy", "dendritic"]
Diff = Literal["front_ahead", "front_behind", "rougher", "smoother", "more_branches", "fewer_branches",
               "detached_fragments", "blurred_interface", "other"]


class CompareReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    morphology_moose: Morph
    morphology_surrogate: List[Morph]
    same_morphology: bool
    differences: List[Diff]
    artifacts: List[str]
    similarity: int = Field(ge=1, le=5)
    evidence: List[str] = Field(min_length=1)
    summary: str


class BlindReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    simulation_row: Literal["A", "B"]
    confidence: int = Field(ge=1, le=5)
    reason: str


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    evidence: List[str] = Field(min_length=1)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overall: Literal["reliable", "reliable_with_limits", "not_reliable"]
    findings: List[Finding] = Field(min_length=1)
    failure_modes: List[str]
    recommendations: List[str]
    summary: str


def ask_qwen(a, prompt, images, schema_cls, record_path):
    """One Ollama chat request with a JSON schema; returns the validated answer or None."""
    body = {"model": a.qwen_model, "stream": False, "format": schema_cls.model_json_schema(),
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(i).decode() for i in images]}],
            "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 4096}}
    req = urllib.request.Request(f"{a.host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=a.timeout) as r:
        raw = json.loads(r.read())
    content = raw.get("message", {}).get("content", "")
    try:
        answer, error = schema_cls.model_validate_json(content), None
    except ValidationError as e:
        answer, error = None, str(e)
    save_json({"model": a.qwen_model, "prompt": prompt, "raw": content, "error": error,
               "answer": answer.model_dump() if answer else None}, record_path)
    return answer


def eta_panel(rows, labels, times):
    """PNG bytes: one row per eta sequence (n, H, W), fixed color scale."""
    n = len(times)
    fig, axes = plt.subplots(len(rows), n, figsize=(1.9 * n + 0.6, 1.9 * len(rows)), squeeze=False)
    for r, (seq, lab) in enumerate(zip(rows, labels)):
        for j in range(n):
            ax = axes[r, j]
            ax.imshow(seq[j], origin="lower", cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={times[j]:.1f}", fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def check_compare(rep, r, a):
    """Agreement of Qwen's answers with the metrics; None where the metrics give no clear answer."""
    out = {}
    if r["gt_class"] != "none":
        out["morphology_moose"] = MORPH_TO_CLASS[rep.morphology_moose] == r["gt_class"]
    shown = r["sample_classes"][:len(rep.morphology_surrogate)]
    if len(rep.morphology_surrogate) == min(a.qwen_rows, len(r["sample_classes"])):
        out["morphology_surrogate"] = all(MORPH_TO_CLASS[m] == c for m, c in zip(rep.morphology_surrogate, shown))
    d = set(rep.differences)
    fd, ad = r["final_front_diff"], r["final_amp_diff"]
    out["front_ahead"] = ("front_ahead" in d) == (fd > a.front_tol)
    out["front_behind"] = ("front_behind" in d) == (fd < -a.front_tol)
    out["rougher"] = ("rougher" in d) == (ad > a.amp_tol)
    out["smoother"] = ("smoother" in d) == (ad < -a.amp_tol)
    out["detached_fragments"] = ("detached_fragments" in d) == (r["max_sample_detached"] > 0.01 >= r["gt_detached"])
    return out


def flatten(d, prefix=""):
    keys = set()
    for k, v in d.items():
        p = f"{prefix}{k}"
        keys.add(p)
        if isinstance(v, dict):
            keys |= flatten(v, p + ".")
    return keys


def run_qwen(results, summary, a, out_dir):
    case_dir = os.path.join(out_dir, "cases")
    qdir = os.path.join(out_dir, "qwen")
    os.makedirs(qdir, exist_ok=True)
    checks, blind = {}, {}
    try:
        for i, r in enumerate(results):
            p = np.load(os.path.join(case_dir, f"{r['case']}_panel.npz"))
            times, gt, samples = p["times"], p["gt"], p["samples"]
            k = min(a.qwen_rows, len(samples))
            rows = [f"S{j + 1}" for j in range(k)]
            img = eta_panel([gt, *samples[:k]], ["MOOSE", *rows], times)
            with open(os.path.join(qdir, f"{r['case']}_compare.png"), "wb") as f:
                f.write(img)
            tstr = ", ".join(f"{t:.1f}" for t in times)
            rep = ask_qwen(a, COMPARE_PROMPT.format(times=tstr, rows=", ".join(rows), nrows=k, lang=a.lang), [img],
                           CompareReport, os.path.join(qdir, f"{r['case']}_compare.json"))
            if rep:
                checks[r["case"]] = check_compare(rep, r, a)

            flip = random.Random(f"{a.seed}:{r['case']}").random() < 0.5  # True: simulation is row B
            pair = [samples[0], gt] if flip else [gt, samples[0]]
            img = eta_panel(pair, ["A", "B"], times)
            with open(os.path.join(qdir, f"{r['case']}_blind.png"), "wb") as f:
                f.write(img)
            rep = ask_qwen(a, BLIND_PROMPT.format(times=tstr, lang=a.lang), [img], BlindReport,
                           os.path.join(qdir, f"{r['case']}_blind.json"))
            if rep:
                truth = "B" if flip else "A"
                blind[r["case"]] = {"answer": rep.simulation_row, "truth": truth,
                                    "correct": rep.simulation_row == truth, "confidence": rep.confidence}
            print(f"qwen {i + 1}/{len(results)} {r['case']}: compare {checks.get(r['case'])}, blind {blind.get(r['case'])}")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        print(f"[WARN] Ollama request failed ({e}); is `ollama serve` running with {a.qwen_model}? "
              f"Numerical results are kept; rerun with --qwen-only.")
        return None

    votes = collections.defaultdict(list)
    for c in checks.values():
        for k, v in c.items():
            votes[k].append(v)
    reading = {
        "metric_agreement": {k: float(np.mean(v)) for k, v in votes.items()},
        "metric_agreement_overall": float(np.mean([x for v in votes.values() for x in v])) if votes else None,
        "blind_accuracy": float(np.mean([b["correct"] for b in blind.values()])) if blind else None,
        "blind_cases": len(blind), "compare_cases": len(checks),
        "note": "blind accuracy near 0.5 means Qwen cannot tell the surrogate from MOOSE",
    }
    save_json({"reading": reading, "per_case": checks, "blind": blind}, os.path.join(qdir, "reading_checks.json"))

    # Keep the request short: per-case rows only for the worst cases.
    per_case = summary["per_case"]
    worst = sorted(per_case, key=lambda n: (per_case[n]["sample_classes"].count(per_case[n]["gt_class"]),
                                            -abs(per_case[n]["final_front_diff"])))[:20]
    evidence = {**{k: v for k, v in summary.items() if k not in ("horizon_error", "per_case")},
                "per_case": {n: per_case[n] for n in worst}, "qwen_reading": reading}
    with open(os.path.join(out_dir, "summary.png"), "rb") as f:
        fig = f.read()
    try:
        verdict = ask_qwen(a, VERDICT_PROMPT.format(lang=a.lang, evidence=json.dumps(tolist(evidence), indent=1)),
                           [fig], Verdict, os.path.join(qdir, "verdict_request.json"))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        print(f"[WARN] verdict request failed: {e}")
        verdict = None
    review = None
    if verdict:
        known = flatten(tolist(evidence))
        # Qwen sometimes writes "key = value"; the key is the part before "=".
        unknown = sorted({e for f in verdict.findings for e in f.evidence
                          if e.split("=")[0].strip().strip("'\"`") not in known})
        evaluated = [c for c in summary["criteria"].values() if c["passed"] is not None]
        crit = summary["criteria"]
        if all(c["passed"] for c in evaluated):
            expected = "reliable"
        elif any(crit[n]["passed"] is False for n in crit if n == "class_agreement" or n.startswith("seed_ratio")):
            expected = "not_reliable"
        else:
            expected = "reliable_with_limits"
        review = {"unknown_evidence_keys": unknown, "expected_overall": expected,
                  "overall_consistent": verdict.overall == expected}
        save_json({"verdict": verdict.model_dump(), "review": review}, os.path.join(qdir, "verdict.json"))
    write_report(summary, reading, verdict, review, a, out_dir)
    return reading


def write_report(summary, reading, verdict, review, a, out_dir):
    L = [f"# Surrogate validation: {os.path.basename(os.path.normpath(a.run))}", "",
         f"Cases: {len(summary['cases'])}, samples per case: {summary['samples']}, guidance: {summary['guidance']}", ""]
    if summary["leakage_cases"]:
        L += [f"Warning: these cases include training frames: {', '.join(summary['leakage_cases'])}", ""]
    L += ["## Criteria (computed by code)", "", "| criterion | value | threshold | passed |", "|---|---|---|---|"]
    for n, c in summary["criteria"].items():
        v = "n/a" if c["value"] is None else f"{c['value']:.3f}"
        L.append(f"| {n} | {v} | {c['threshold']} | {'not evaluated' if c['passed'] is None else c['passed']} |")
    ht = summary["trusted_horizon_time"]
    L += ["", f"Trusted horizon: {summary['trusted_horizon_frames']} frames" + (f" ({ht:g} time units)" if ht else ""),
          f"Seed floor: {summary['seed_floor'].get('groups', 0)} replicate groups. "
          f"{summary['seed_floor'].get('note', '')}", "", "![summary](summary.png)", ""]
    if reading:
        L += ["## Qwen reading checks", "",
              f"- agreement of Qwen's answers with the metrics: {reading['metric_agreement_overall']}",
              f"- by question: {json.dumps(reading['metric_agreement'])}",
              f"- blind test accuracy (0.5 = cannot tell): {reading['blind_accuracy']} over {reading['blind_cases']} cases", ""]
    if verdict:
        L += [f"## Qwen verdict (draft, {a.qwen_model})", "",
              f"Overall: **{verdict.overall}** (consistent with the criteria: {review['overall_consistent']}; "
              f"expected {review['expected_overall']})", ""]
        if review["unknown_evidence_keys"]:
            L += [f"Unknown evidence keys (claims not supported): {', '.join(review['unknown_evidence_keys'])}", ""]
        L += [verdict.summary, "", "Findings:"] + [f"- {f.claim} [{', '.join(f.evidence)}]" for f in verdict.findings]
        L += ["", "Failure modes:"] + [f"- {x}" for x in verdict.failure_modes]
        L += ["", "Recommendations:"] + [f"- {x}" for x in verdict.recommendations]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------- main

def load_saved(out_dir):
    with open(os.path.join(out_dir, "summary.json")) as f:
        summary = json.load(f)
    results = []
    for name in summary["cases"]:
        with open(os.path.join(out_dir, "cases", f"{name}.json")) as f:
            results.append(json.load(f))
    return results, summary


def main(argv=None):
    a = parse_args(argv)
    out_dir = a.output or os.path.join(a.run, "validation")
    case_out = os.path.join(out_dir, "cases")
    os.makedirs(case_out, exist_ok=True)

    if a.qwen_only:
        results, summary = load_saved(out_dir)
    else:
        ctx = load_run(a)
        names = a.cases or list(ctx.cfg.get("test_ranges", {}))
        if not names:
            raise ValueError("the run has no test cases; pass --cases")
        dirs = discover_cases(ctx.data, names)
        results = [r for n, d in dirs.items() if (r := evaluate_case(n, d, ctx, a, case_out)) is not None]
        if not results:
            raise ValueError("no case could be evaluated")
        floor = seed_floor(ctx, a)
        sweeps = [sweep(p, results, ctx, a, out_dir) for p in a.sweep]
        summary = summarize(results, floor, sweeps, a, ctx)
        save_json(tolist(summary), os.path.join(out_dir, "summary.json"))
        plot_summary(summary, os.path.join(out_dir, "summary.png"))
        for n, c in summary["criteria"].items():
            v = "n/a" if c["value"] is None else f"{c['value']:.3f}"
            print(f"  {n:24s} {v:>8s}  {c['threshold']:28s} "
                  f"{'not evaluated' if c['passed'] is None else ('PASS' if c['passed'] else 'FAIL')}")
        print(f"  trusted horizon: {summary['trusted_horizon_frames']} frames")
        results = [tolist(r) for r in results]
        del ctx
        torch.cuda.empty_cache()  # free the GPU for Ollama

    if a.no_qwen or run_qwen(results, summary, a, out_dir) is None:
        write_report(summary, None, None, None, a, out_dir)
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
