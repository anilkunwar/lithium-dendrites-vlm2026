#!/usr/bin/env python3
"""Let Qwen explore the parameter space through the frame-diffusion surrogate.

Qwen plans experiments; code runs them on the surrogate, measures the morphology and fits the
boundary; Qwen reads the results and plans the next step. Two kinds of question:

  region    where in the plane of one or two parameters (--axes) is the goal met? The goal is a
            condition on the final-frame metrics (step1's front metrics and class, and the onset time).
  concept   is a combination of parameters (an expression such as "L2o*abs(POT_LEFT)/L1o") what controls
            the outcome? test_invariance places points on lines where the expression is constant: if
            the expression is the controlling quantity, the outcome stays the same along each line
            and changes between lines. Its separation (AUC) is compared with each parameter alone.

Every point is evaluated with --samples rollouts from the same start frame and the same noise seeds
(common random numbers, so differences between points come from the parameters). The fraction of
samples that meet the goal is the point's score p. Qwen sees a table of all points, the fitted
boundary (logistic regression in log10 or linear coordinates) and a phase plot, then answers in JSON
with one action:

  grid            n x n points over the axes (optionally a sub-range)
  points          explicit points
  refine          k points chosen by code near the fitted boundary, away from existing points
  test_invariance expression, number of levels and points per level
  propose_moose   points worth checking with MOOSE (written to moose_candidates.json; nothing is run)
  finish          stop and write the final report

Only parameters inside the training range are allowed, and the rollout length defaults to the trusted
horizon of validate_surrogate.py. Conclusions are drafts: Qwen must cite point (P..) and experiment (E..)
ids, which are checked; MOOSE candidates should be run before a result is reported.

Run from vision-language-model/:
python frame_diffusion/explore_agent.py --run runs/<run> --base-case case_005 --axes POT_LEFT L2o \
    --goal-file goals/uniform_growth.json
python frame_diffusion/explore_agent.py --run runs/<run> --base-case case_005 --axes POT_LEFT L2o \
    --target-class uniform --min-front 0.3 --budget 120 --samples 4
python frame_diffusion/explore_agent.py --resume runs/<run>/explore/<dir>   # continue a session

Goal file (JSON); condition keys: class, onset_time, front_mean, front_max, amplitude, roughness,
length_ratio, detached_area; each with "in", "min" and/or "max":
{"description": "flat, dense deposit that still grows",
 "conditions": {"class": {"in": ["uniform"]}, "front_mean": {"min": 0.3}}}
"""

from __future__ import annotations

import argparse
import ast
import collections
import datetime
import io
import json
import math
import operator
import os
import urllib.error
from typing import Dict, List, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import brentq
from scipy.stats import rankdata

from data import discover_cases, load_case, load_params, save_json
from validate_surrogate import (METRICS, ask_qwen, case_params, encode_params, load_run, sample_rollouts, series,
                                tolist)

GOAL_KEYS = (*METRICS, "class", "onset_time")
TABLE_ROWS = 80  # points listed to Qwen; the rest are summarized


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="training output directory")
    p.add_argument("--resume", help="session directory to continue (its settings are reused)")
    p.add_argument("--data", help="data root (default: the one used in training)")
    p.add_argument("--base-case", help="case that supplies the start frames and the fixed parameters")
    p.add_argument("--t0", type=float, help="start after this time of the base case (default: its first frames)")
    p.add_argument("--axes", nargs="+", help="one or two parameters to explore")
    p.add_argument("--goal-file", help="JSON with description and conditions (see above)")
    p.add_argument("--target-class", nargs="*", help="shortcut: final class in these (uniform/transitional/dendritic)")
    p.add_argument("--min-front", type=float, help="shortcut: final front_mean at least this")
    p.add_argument("--max-amplitude", type=float, help="shortcut: final amplitude at most this")
    p.add_argument("--num-frames", type=int, help="rollout length (default: trusted horizon from validation)")
    p.add_argument("--dt", type=float, help="time step beyond the base case data (default: the training --dt)")
    p.add_argument("--samples", type=int, default=4, help="rollouts per point")
    p.add_argument("--steps", type=int, default=25, help="DDIM steps")
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.5, help="eta threshold for the deposit")
    p.add_argument("--li-ratio", type=float, default=76.9)
    p.add_argument("--budget", type=int, default=150, help="maximum number of evaluated points")
    p.add_argument("--max-iters", type=int, default=15)
    p.add_argument("--max-points-per-step", type=int, default=36)
    p.add_argument("--qwen-model", default="qwen3-vl:8b-instruct")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--no-images", action="store_true", help="send tables only (for text-only models)")
    p.add_argument("--lang", default="English", help="language of Qwen's text fields")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--output", help="default: <run>/explore/<timestamp>")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.resume:
        with open(os.path.join(a.resume, "session.json")) as f:
            saved = json.load(f)["args"]
        p.set_defaults(**{k: v for k, v in saved.items() if k not in ("resume", "output", "max_iters", "budget")})
        a = p.parse_args(argv)
        a.output = a.resume
    if not a.run or not a.base_case or not a.axes:
        p.error("--run, --base-case and --axes are required (or --resume)")
    if len(a.axes) > 2:
        p.error("--axes takes one or two parameters")
    return a


# ---------------------------------------------------------------- goal and expressions

def load_goal(a):
    if a.goal_file:
        with open(a.goal_file) as f:
            goal = json.load(f)
    else:
        cond = {}
        if a.target_class:
            cond["class"] = {"in": a.target_class}
        if a.min_front is not None:
            cond["front_mean"] = {"min": a.min_front}
        if a.max_amplitude is not None:
            cond["amplitude"] = {"max": a.max_amplitude}
        goal = {"description": "; ".join(f"{k} {v}" for k, v in cond.items()), "conditions": cond}
    if not goal.get("conditions"):
        raise ValueError("no goal: pass --goal-file or --target-class / --min-front / --max-amplitude")
    bad = [k for k in goal["conditions"] if k not in GOAL_KEYS]
    if bad:
        raise ValueError(f"unknown goal keys {bad}; allowed: {GOAL_KEYS}")
    return goal


def meets(m, conditions):
    for k, c in conditions.items():
        v = m.get(k)
        if "in" in c and v not in c["in"]:
            return False
        x = math.inf if v is None else v  # onset_time None: never became unstable
        if "min" in c and not x >= c["min"]:
            return False
        if "max" in c and not x <= c["max"]:
            return False
    return True


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow}
_FUNCS = {"abs": abs, "log": math.log, "log10": math.log10, "exp": math.exp, "sqrt": math.sqrt}


def eval_expr(expr, values):
    """Arithmetic on parameter names only (no attribute access, no other calls)."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id not in values:
                raise ValueError(f"unknown name {n.id!r}; parameters: {sorted(values)}")
            return float(values[n.id])
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponents are limited to |p| <= 10")
            return _OPS[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS
                and len(n.args) == 1 and not n.keywords):
            return float(_FUNCS[n.func.id](ev(n.args[0])))
        raise ValueError(f"not allowed in expressions: {type(n).__name__}")
    return ev(ast.parse(expr, mode="eval"))


def auc(score, label):
    """Probability that a random goal point scores higher than a random other point (None if one class)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return None
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------- coordinates and boundary

class Space:
    """Axis bounds from the training cases; log10 coordinates for log-scaled parameters."""

    def __init__(self, axes, ctx, data_root):
        dirs = discover_cases(data_root)
        train = [n for n in ctx.cfg.get("train_ranges", {}) if n in dirs]
        if not train:
            raise ValueError("no training cases found under the data root; pass --data")
        names, raw = load_params({n: dirs[n] for n in train}, ctx.param_norm.names)
        self.axes = list(axes)
        self.log = [ax in ctx.param_norm.log for ax in self.axes]
        self.lo, self.hi = [], []
        for ax, lg in zip(self.axes, self.log):
            v = np.array([raw[n][names.index(ax)] for n in train])
            if lg and (v <= 0).any():
                raise ValueError(f"{ax} is log-scaled but has non-positive training values")
            self.lo.append(float(v.min()))
            self.hi.append(float(v.max()))

    def to_x(self, i, v):  # parameter value -> fit coordinate
        return math.log10(v) if self.log[i] else v

    def from_x(self, i, x):
        return 10 ** x if self.log[i] else x

    def to_u(self, point):  # dict -> [0, 1]^d
        return np.array([(self.to_x(i, point[ax]) - self.to_x(i, self.lo[i])) /
                         max(self.to_x(i, self.hi[i]) - self.to_x(i, self.lo[i]), 1e-12)
                         for i, ax in enumerate(self.axes)])

    def from_u(self, u):
        return {ax: float(self.from_x(i, self.to_x(i, self.lo[i]) + u[i] * (self.to_x(i, self.hi[i]) -
                                                                            self.to_x(i, self.lo[i]))))
                for i, ax in enumerate(self.axes)}

    def inside(self, point, rtol=1e-6):
        """Within the training range (a small tolerance absorbs the log10 round trip)."""
        return all(self.lo[i] - rtol * max(1.0, abs(self.lo[i])) <= point[ax] <= self.hi[i] + rtol * max(1.0, abs(self.hi[i]))
                   for i, ax in enumerate(self.axes))

    def describe(self):
        return [f"{ax}: [{self.lo[i]:.4g}, {self.hi[i]:.4g}] ({'log10' if self.log[i] else 'linear'} axis)"
                for i, ax in enumerate(self.axes)]


def fit_logistic(U, y, ridge=1e-2, iters=100):
    """Logistic regression with soft labels y in [0, 1] (Newton steps); returns weights [w..., b]."""
    A = np.hstack([U, np.ones((len(U), 1))])
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - y) + ridge * w
        H = A.T @ (A * (p * (1 - p) + 1e-6)[:, None]) + ridge * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.linalg.norm(step) < 1e-8:
            break
    return w


def boundary(points, space):
    """Fitted boundary p = 0.5 and its description in parameter terms."""
    if len(points) < 4:
        return None
    U = np.stack([space.to_u(p["axes"]) for p in points])
    y = np.array([p["p"] for p in points])
    if y.min() >= 0.5 or y.max() < 0.5:
        return {"note": "all points are on one side of the goal; no boundary yet", "w": None}
    w = fit_logistic(U, y)
    pred = 1 / (1 + np.exp(-(np.hstack([U, np.ones((len(U), 1))]) @ w)))
    out = {"w": w.tolist(), "accuracy": float(np.mean((pred >= 0.5) == (y >= 0.5)))}
    names = [("log10 " if space.log[i] else "") + ax for i, ax in enumerate(space.axes)]
    if len(space.axes) == 1:
        if abs(w[0]) > 1e-9:
            u0 = -w[1] / w[0]
            trend = f"it {'rises' if w[0] > 0 else 'falls'} with {space.axes[0]}"
            if 0 <= u0 <= 1:
                out["crossing"] = space.from_u([u0])[space.axes[0]]
                out["text"] = f"goal probability crosses 0.5 at {space.axes[0]} = {out['crossing']:.4g}; {trend}"
            else:
                out["text"] = f"the fitted crossing lies outside the explored range; {trend}"
    elif abs(w[1]) > 1e-9:
        # Slope of the boundary in fit coordinates: d x2 / d x1.
        s = [space.to_x(i, space.hi[i]) - space.to_x(i, space.lo[i]) for i in range(2)]
        out["slope"] = float(-(w[0] / s[0]) * (s[1] / w[1]))
        table = []
        for u1 in np.linspace(0, 1, 5):
            u2 = -(w[2] + w[0] * u1) / w[1]
            if 0 <= u2 <= 1:
                pt = space.from_u([u1, u2])
                table.append({ax: round(v, 6) for ax, v in pt.items()})
        out["boundary_points"] = table
        law = (f"; with both axes in log10 this is a power law {space.axes[1]} ~ {space.axes[0]}^{out['slope']:.2f}"
               if all(space.log) else "")
        out["text"] = (f"boundary: d({names[1]})/d({names[0]}) = {out['slope']:.3g}{law}. "
                       f"Goal side: weights {w[0]:+.2f} on {names[0]}, {w[1]:+.2f} on {names[1]} (normalized axes)")
    return out


# ---------------------------------------------------------------- Qwen schema

Tool = Literal["grid", "points", "refine", "test_invariance", "propose_moose", "finish"]


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Tool
    n: List[int] = Field(default_factory=list, description="grid: points per axis")
    ranges: Dict[str, List[float]] = Field(default_factory=dict, description="grid: optional [lo, hi] per axis")
    points: List[Dict[str, float]] = Field(default_factory=list, description="points / propose_moose")
    k: int = Field(default=0, description="refine: number of points; test_invariance: points per level")
    expr: str = Field(default="", description="test_invariance: expression of parameter names")
    levels: int = Field(default=3, description="test_invariance: number of constant-expression lines")


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation: str
    hypothesis: str
    action: Action
    conclusion_so_far: str
    confidence: int = Field(ge=1, le=5)
    cites: List[str]


class Concept(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expr: str
    verdict: Literal["supported", "rejected", "untested"]
    evidence: List[str]


class FinalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    region: str
    boundary: str
    concepts: List[Concept]
    moose_points: List[Dict[str, float]]
    caveats: List[str]
    summary: str
    cites: List[str]


SYSTEM = """You are a researcher exploring a phase-field model of lithium electrodeposition through a fast
neural-network surrogate. The deposit grows from the left edge; final-frame metrics are fractions of the domain:
front_mean (mean front position), amplitude (front spread across y), length_ratio (interface length / height,
1 = flat), detached_area; class is uniform / transitional / dendritic (none = no attached deposit); onset_time is
the first time the front is not uniform (null = never).

Goal: {goal}
Goal conditions on the final frame: {conditions}
Explored parameters and their allowed (training) ranges: {axes}
Fixed parameters (from {base}): {fixed}
Each point: {samples} rollouts of {frames} frames from the same start frame with the same noise seeds;
p = fraction of rollouts meeting the goal. Budget: {used}/{budget} points used.

Tools (choose exactly one per step):
- grid: n = [points per axis] (one value per explored parameter), optional ranges {{"param": [lo, hi]}}.
- points: explicit points, e.g. [{{"{ax0}": value, ...}}] with every explored parameter.
- refine: k points that code places near the fitted boundary, away from existing points.
- test_invariance: expr = a combination of parameter names (+ - * / **, abs, log, log10, exp, sqrt; fixed
  parameters may appear), levels = number of lines where expr is constant, k = points per line. Supported if the
  outcome is constant along each line and its AUC beats every single parameter.
- propose_moose: points to verify with the real simulation (no budget cost). Choose points on both sides of the
  boundary and points that support or reject a concept.
- finish: stop when the boundary is located within the point spacing, or the budget is nearly used.
Rules: use only values inside the allowed ranges; the surrogate can be wrong, so weigh p near 0.5 and large std
as uncertain; base statements on the table, cite point ids (P..) and experiment ids (E..) in cites.
Write observation, hypothesis and conclusion_so_far in {lang}.
"""


# ---------------------------------------------------------------- session

class Session:
    def __init__(self, a):
        self.a = a
        self.ctx = load_run(a)
        if not self.ctx.param_norm:
            raise ValueError("the model is not conditioned on parameters")
        unknown = [ax for ax in a.axes if ax not in self.ctx.param_norm.names]
        if unknown:
            raise ValueError(f"{unknown} are not model inputs; the model uses {self.ctx.param_norm.names}")
        if a.dt:
            self.ctx.roll.dt = a.dt
        self.goal = load_goal(a)
        data_root = self.ctx.data
        self.space = Space(a.axes, self.ctx, data_root)
        case_dir = discover_cases(data_root, [a.base_case])[a.base_case]
        self.frames, self.times = load_case(case_dir, self.ctx.res, self.ctx.dt)
        K = self.ctx.K
        self.start = K if a.t0 is None else max(K, int(np.searchsorted(self.times, a.t0, side="right")))
        self.base_params = case_params(self.ctx, a.base_case, case_dir)
        available = len(self.times) - self.start
        if a.num_frames:
            self.n = a.num_frames
        else:
            self.n = self.trusted_horizon()
            if self.n > available and not self.ctx.roll.dt:
                print(f"[INFO] {a.base_case} has {available} frames after the start; rollout length cut from "
                      f"{self.n} to {available} (pass --num-frames, or --dt to go beyond the data)")
                self.n = available
        if self.n > available and not self.ctx.roll.dt:
            raise ValueError(f"--num-frames {self.n} goes past the {available} frames of {a.base_case} after the "
                             f"start; use --num-frames <= {available}, or pass --dt for the spacing beyond the data")
        if self.n * a.samples * a.steps > 20000:
            print(f"[WARN] each point needs {self.n} frames x {a.samples} samples x {a.steps} steps = "
                  f"{self.n * a.samples * a.steps} model evaluations; consider a smaller --num-frames")
        self.points, self.experiments, self.eta, self.moose, self.iteration = [], [], {}, [], 0

    def trusted_horizon(self):
        path = os.path.join(self.a.run, "validation", "summary.json")
        if os.path.exists(path):
            with open(path) as f:
                h = json.load(f).get("trusted_horizon_frames")
            if h:
                print(f"rollout length = trusted horizon from validation: {h} frames")
                return int(h)
        raise ValueError("no trusted horizon (run validate_surrogate.py first) and no --num-frames")

    # ------------------------------------------------ evaluation

    def evaluate(self, axes_values, expt):
        """Evaluate one point; returns its record (or the existing one for a repeated point)."""
        for p in self.points:
            if all(math.isclose(p["axes"][ax], axes_values[ax], rel_tol=1e-6) for ax in self.space.axes):
                return p
        values = {**self.base_params, **axes_values}
        runs = sample_rollouts(self.ctx, self.frames, self.times, self.start, self.n,
                               encode_params(self.ctx, values), self.a.samples, self.a.seed)
        t = np.array([self.times[i] if i < len(self.times) else self.times[-1] + (i - len(self.times) + 1) *
                      self.ctx.roll.dt for i in range(self.start, self.start + self.n)])
        finals = []
        for run in runs:
            s = series(run, self.a)
            m = {k: float(s[k][-1]) for k in METRICS}
            m["class"] = s["class"][-1]
            m["onset_time"] = next((float(tt) for c, tt in zip(s["class"], t) if c in ("transitional", "dendritic")),
                                   None)
            finals.append(m)
        ok = [meets(m, self.goal["conditions"]) for m in finals]
        onsets = [m["onset_time"] for m in finals if m["onset_time"] is not None]
        rec = {
            "id": f"P{len(self.points) + 1}", "experiment": expt, "axes": axes_values, "p": float(np.mean(ok)),
            "classes": dict(collections.Counter(m["class"] for m in finals)),
            **{f"{k}_mean": float(np.mean([m[k] for m in finals])) for k in ("front_mean", "amplitude", "length_ratio")},
            "front_std": float(np.std([m["front_mean"] for m in finals])),
            "onset_mean": float(np.mean(onsets)) if onsets else None, "onset_count": len(onsets),
            "final_time": float(t[-1]),
        }
        self.points.append(rec)
        self.eta[rec["id"]] = runs[0, -1, 0].astype(np.float16)
        return rec

    def budget_left(self):
        return self.a.budget - len(self.points)

    def run_points(self, pts, expt, note):
        evaluated, rejected = [], []
        for pt in pts:
            if self.budget_left() <= 0:
                note.append("budget exhausted; remaining points skipped")
                break
            if set(pt) != set(self.space.axes) or not self.space.inside(pt):
                rejected.append(pt)
                continue
            evaluated.append(self.evaluate({ax: float(pt[ax]) for ax in self.space.axes}, expt)["id"])
        if rejected:
            note.append(f"{len(rejected)} points rejected (missing axes or outside the training range): {rejected[:5]}")
        return evaluated

    # ------------------------------------------------ tools

    def grid(self, act, expt, note):
        d = len(self.space.axes)
        n = list(act.n[:d]) or [5]
        n += [n[0]] * (d - len(n))
        axes_u = []
        for i, ax in enumerate(self.space.axes):
            lo, hi = 0.0, 1.0
            if ax in act.ranges and len(act.ranges[ax]) == 2:
                r = sorted(act.ranges[ax])
                lo_pt, hi_pt = self.space.from_u([0.0] * d), self.space.from_u([1.0] * d)
                r = [min(max(r[0], lo_pt[ax]), hi_pt[ax]), min(max(r[1], lo_pt[ax]), hi_pt[ax])]
                lo = self.space.to_u({**lo_pt, ax: r[0]})[i]
                hi = self.space.to_u({**lo_pt, ax: r[1]})[i]
            axes_u.append(np.linspace(lo, hi, max(2, n[i])))
        mesh = np.stack(np.meshgrid(*axes_u, indexing="ij"), -1).reshape(-1, d)
        if len(mesh) > self.a.max_points_per_step:
            note.append(f"grid of {len(mesh)} points truncated to {self.a.max_points_per_step}")
            mesh = mesh[np.linspace(0, len(mesh) - 1, self.a.max_points_per_step).astype(int)]
        return self.run_points([self.space.from_u(u) for u in mesh], expt, note)

    def refine(self, act, expt, note):
        k = max(1, min(act.k or 6, self.a.max_points_per_step))
        rng = np.random.default_rng(self.a.seed + len(self.points))
        cand = rng.random((2000, len(self.space.axes)))
        have = [self.space.to_u(p["axes"]) for p in self.points]
        b = boundary(self.points, self.space)
        if b and b.get("w"):
            w = np.array(b["w"])
            q = 1 / (1 + np.exp(-(np.hstack([cand, np.ones((len(cand), 1))]) @ w)))
            closeness = 1 - 2 * np.abs(q - 0.5)
        else:
            closeness = np.zeros(len(cand))
            note.append("no boundary fitted yet; refine placed space-filling points")
        chosen = []
        for _ in range(k):
            ref = np.array(have + chosen) if (have or chosen) else None
            spread = np.min(np.linalg.norm(cand[:, None] - ref[None], axis=-1), axis=1) if ref is not None \
                else np.ones(len(cand))
            j = int(np.argmax(closeness + 2.0 * spread))
            chosen.append(cand[j])
        return self.run_points([self.space.from_u(u) for u in chosen], expt, note)

    def test_invariance(self, act, expt, note):
        if not act.expr:
            note.append("test_invariance needs expr")
            return [], None
        fixed = {k: v for k, v in self.base_params.items() if k not in self.space.axes}

        def pi(pt):
            return eval_expr(act.expr, {**fixed, **pt})

        def pi_or_nan(pt):
            try:
                return pi(pt)
            except (ZeroDivisionError, OverflowError, ValueError):
                return math.nan

        try:  # a syntax error or an unknown name rejects the expression
            pi(self.space.from_u([0.5] * len(self.space.axes)))
        except (ValueError, SyntaxError) as e:
            note.append(f"expression rejected: {e}")
            return [], None
        existing = [v for v in (pi_or_nan(p["axes"]) for p in self.points) if np.isfinite(v)]
        new_ids = []
        if len(self.space.axes) == 2:
            levels = max(2, min(act.levels or 3, 6))
            per = max(2, min(act.k or 4, 8))
            ref = existing or [pi(self.space.from_u(u)) for u in np.random.default_rng(0).random((200, 2))]
            targets = np.quantile(ref, np.linspace(0.2, 0.8, levels))
            pts = []
            for lv in targets:
                for u1 in np.linspace(0.05, 0.95, per):
                    f = lambda u2: pi(self.space.from_u([u1, u2])) - lv  # noqa: E731
                    try:
                        if f(0.0) * f(1.0) < 0:
                            pts.append((float(lv), self.space.from_u([u1, brentq(f, 0.0, 1.0)])))
                    except (ValueError, ZeroDivisionError, OverflowError):
                        pass
            if not pts:
                note.append("no constant-expression line crosses the explored range")
            level_of = {}
            for lv, pt in pts[:self.a.max_points_per_step]:
                ids = self.run_points([pt], expt, note)
                if ids:
                    new_ids.append(ids[0])
                    level_of[ids[0]] = lv
        else:
            note.append("with one axis, only the separation (AUC) is computed")
            level_of = {}
        # Separation of the goal by the expression and by each parameter alone, over all points.
        vals = np.array([pi_or_nan(p["axes"]) for p in self.points])
        ok = np.isfinite(vals)
        label = np.array([p["p"] >= 0.5 for p in self.points])
        score = {"expr": act.expr, "n_points": int(ok.sum())}
        a_expr = auc(vals[ok], label[ok])
        score["auc_expr"] = None if a_expr is None else max(a_expr, 1 - a_expr)
        for ax in self.space.axes:
            a_ax = auc(np.array([p["axes"][ax] for p in self.points])[ok], label[ok])
            score[f"auc_{ax}"] = None if a_ax is None else max(a_ax, 1 - a_ax)
        if level_of:
            by_level = collections.defaultdict(list)
            for p in self.points:
                if p["id"] in level_of:
                    by_level[level_of[p["id"]]].append(p["p"] >= 0.5)
            # 1.0: every line has one outcome; 0.5: lines are mixed.
            score["line_consistency"] = float(np.mean([max(np.mean(v), 1 - np.mean(v)) for v in by_level.values()]))
            score["line_outcomes"] = {f"{lv:.4g}": f"{sum(v)}/{len(v)} meet the goal" for lv, v in by_level.items()}
        singles = [v for k, v in score.items() if k.startswith("auc_") and k != "auc_expr" and v is not None]
        if score["auc_expr"] is not None and singles:
            score["beats_single_parameters"] = bool(score["auc_expr"] > max(singles) + 0.02)
        return new_ids, score

    def propose_moose(self, act, expt, note):
        added = []
        for pt in act.points:
            if set(pt) != set(self.space.axes) or not self.space.inside(pt):
                note.append(f"MOOSE point rejected (axes or range): {pt}")
                continue
            self.moose.append({"id": f"M{len(self.moose) + 1}", "experiment": expt,
                               "params": {**self.base_params, **{k: float(v) for k, v in pt.items()}}})
            added.append(self.moose[-1]["id"])
        return added

    def execute(self, act):
        expt = f"E{len(self.experiments) + 1}"
        note, result, ids = [], None, []
        if act.tool == "grid":
            ids = self.grid(act, expt, note)
        elif act.tool == "points":
            ids = self.run_points(act.points[:self.a.max_points_per_step], expt, note)
        elif act.tool == "refine":
            ids = self.refine(act, expt, note)
        elif act.tool == "test_invariance":
            ids, result = self.test_invariance(act, expt, note)
        elif act.tool == "propose_moose":
            ids = self.propose_moose(act, expt, note)
        rec = {"id": expt, "iteration": self.iteration, "action": act.model_dump(), "new": ids, "notes": note,
               "result": result}
        if act.tool in ("grid", "points", "refine", "test_invariance"):
            meet = [p for p in self.points if p["id"] in ids and p["p"] >= 0.5]
            rec["summary"] = f"{len(ids)} new points, {len(meet)} meet the goal (p >= 0.5)"
        self.experiments.append(rec)
        return rec

    # ------------------------------------------------ prompt, figures, state

    def table(self):
        pts = self.points
        if len(pts) > TABLE_ROWS:
            near = sorted(pts, key=lambda p: abs(p["p"] - 0.5))[:TABLE_ROWS // 2]
            recent = pts[-TABLE_ROWS // 2:]
            keep = {p["id"] for p in near + recent}
            pts = [p for p in pts if p["id"] in keep]
        rows = []
        for p in pts:
            ax = " ".join(f"{k}={v:.4g}" for k, v in p["axes"].items())
            cls = " ".join(f"{k}:{v}" for k, v in sorted(p["classes"].items()))
            onset = "never" if p["onset_mean"] is None else f"{p['onset_mean']:.4g} ({p['onset_count']}/{self.a.samples})"
            rows.append(f"{p['id']} [{p['experiment']}] {ax} | p={p['p']:.2f} | {cls} | front {p['front_mean_mean']:.3f}"
                        f"±{p['front_std']:.3f} amp {p['amplitude_mean']:.3f} len {p['length_ratio_mean']:.2f} | onset {onset}")
        omitted = len(self.points) - len(pts)
        return "\n".join(rows) + (f"\n({omitted} points not listed)" if omitted else "") if rows else "(no points yet)"

    def history(self):
        out = []
        for e in self.experiments[-8:]:
            act = {k: v for k, v in e["action"].items() if v not in ([], {}, "", 0) and not (k == "levels" and
                                                                                           e["action"]["tool"] != "test_invariance")}
            line = f"{e['id']}: {json.dumps(act)} -> {e.get('summary', ', '.join(e['new']))}"
            if e["result"]:
                line += f"; result {json.dumps(e['result'])}"
            if e["notes"]:
                line += f"; notes: {'; '.join(e['notes'])}"
            out.append(line)
        return "\n".join(out) or "(none)"

    def prompt(self, final=False):
        fixed = {k: round(v, 6) for k, v in self.base_params.items() if k not in self.space.axes}
        head = SYSTEM.format(goal=self.goal["description"], conditions=json.dumps(self.goal["conditions"]),
                             axes="; ".join(self.space.describe()), base=self.a.base_case, fixed=json.dumps(fixed),
                             samples=self.a.samples, frames=self.n, used=len(self.points), budget=self.a.budget,
                             ax0=self.space.axes[0], lang=self.a.lang)
        b = boundary(self.points, self.space)
        body = (f"\nPoints (id [experiment] parameters | p | classes | final metrics mean±std | onset):\n{self.table()}\n"
                f"\nFitted boundary (code): {json.dumps(b) if b else 'not enough points'}\n"
                f"\nExperiments so far:\n{self.history()}\n"
                f"\nMOOSE candidates: {json.dumps([m['id'] + ' ' + json.dumps({k: m['params'][k] for k in self.space.axes}) for m in self.moose])}\n")
        if final:
            body += (f"\nWrite the final report. region: where the goal is met, in parameter terms. boundary: its "
                     f"location and shape (use the fitted boundary). concepts: each tested expression with verdict "
                     f"and evidence ids. moose_points: up to 8 points (with every explored parameter) that would "
                     f"confirm the conclusions with the real simulation. caveats: limits (surrogate errors, sparse "
                     f"regions, fixed parameters). Write text fields in {self.a.lang}.\n")
        else:
            body += f"\nIteration {self.iteration + 1} of at most {self.a.max_iters}. Choose the next action.\n"
        return head + body

    def phase_plot(self):
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        pts = self.points
        if len(self.space.axes) == 1:
            name = self.space.axes[0]
            x = [p["axes"][name] for p in pts]
            ax.scatter(x, [p["p"] for p in pts], c=[p["p"] for p in pts], cmap="coolwarm_r", vmin=0, vmax=1)
            ax2 = ax.twinx()
            ax2.errorbar(x, [p["front_mean_mean"] for p in pts], [p["front_std"] for p in pts], fmt=".", color="0.5")
            ax2.set_ylabel("final front_mean")
            ax.set_xlabel(name), ax.set_ylabel("p (goal met)")
            for p, xi in zip(pts, x):
                ax.annotate(p["id"], (xi, p["p"]), fontsize=6)
            if self.space.log[0]:
                ax.set_xscale("log")
        else:
            a0, a1 = self.space.axes
            if pts:
                sc = ax.scatter([p["axes"][a0] for p in pts], [p["axes"][a1] for p in pts],
                                c=[p["p"] for p in pts], cmap="coolwarm_r", vmin=0, vmax=1,
                                s=[30 + 400 * p["front_std"] for p in pts], edgecolors="k", linewidths=0.4)
                fig.colorbar(sc, ax=ax, label="p (goal met)")
                for p in pts:
                    ax.annotate(p["id"], (p["axes"][a0], p["axes"][a1]), fontsize=6, xytext=(3, 3),
                                textcoords="offset points")
            b = boundary(pts, self.space)
            if b and b.get("w"):
                w = np.array(b["w"])
                u = np.linspace(0, 1, 100)
                uu1, uu2 = np.meshgrid(u, u)
                q = 1 / (1 + np.exp(-(w[0] * uu1 + w[1] * uu2 + w[2])))
                g0 = [self.space.from_u([x, 0])[a0] for x in u]
                g1 = [self.space.from_u([0, y])[a1] for y in u]
                ax.contour(g0, g1, q, levels=[0.5], colors="k", linestyles="--")
            ax.set_xlabel(a0), ax.set_ylabel(a1)
            ax.set_xlim(self.space.lo[0], self.space.hi[0]), ax.set_ylim(self.space.lo[1], self.space.hi[1])
            if self.space.log[0]:
                ax.set_xscale("log")
            if self.space.log[1]:
                ax.set_yscale("log")
        ax.set_title(f"goal: {self.goal['description'][:60]}\ncolor = p, size = front std, dashed = fitted boundary",
                     fontsize=8)
        fig.tight_layout()
        return fig_bytes(fig)

    def montage(self, k=8):
        """Final eta (sample 1) of the points closest to the boundary and the extremes."""
        if not self.points:
            return None
        order = sorted(self.points, key=lambda p: abs(p["p"] - 0.5))
        pick = order[:k - 2] + [max(self.points, key=lambda p: p["p"]), min(self.points, key=lambda p: p["p"])]
        pick = list({p["id"]: p for p in pick}.values())[:k]
        fig, axes = plt.subplots(1, len(pick), figsize=(1.9 * len(pick), 2.3), squeeze=False)
        for axx, p in zip(axes[0], pick):
            axx.imshow(self.eta[p["id"]].astype(np.float32), origin="lower", cmap="viridis", vmin=0, vmax=1)
            axx.set_xticks([]), axx.set_yticks([])
            axx.set_title(f"{p['id']} p={p['p']:.2f}", fontsize=8)
        fig.suptitle("final eta, first sample (yellow = lithium)", fontsize=8)
        fig.tight_layout()
        return fig_bytes(fig)

    def save(self):
        out = self.a.output
        save_json({"args": vars(self.a), "goal": self.goal, "base_params": self.base_params,
                   "space": {"axes": self.space.axes, "lo": self.space.lo, "hi": self.space.hi, "log": self.space.log},
                   "num_frames": self.n, "start": self.start, "iteration": self.iteration,
                   "points": self.points, "experiments": self.experiments, "moose": self.moose},
                  os.path.join(out, "session.json"))
        np.savez_compressed(os.path.join(out, "final_eta.npz"), **self.eta)
        save_json(self.moose, os.path.join(out, "moose_candidates.json"))

    def load(self):
        with open(os.path.join(self.a.output, "session.json")) as f:
            s = json.load(f)
        self.points, self.experiments, self.moose = s["points"], s["experiments"], s["moose"]
        self.iteration = s["iteration"]
        path = os.path.join(self.a.output, "final_eta.npz")
        if os.path.exists(path):
            self.eta = dict(np.load(path))
        print(f"resumed: {len(self.points)} points, {len(self.experiments)} experiments, iteration {self.iteration}")


def fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------- main loop

def ask(session, final, path):
    images = []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not session.a.no_images:
        images = [session.phase_plot()] + ([m] if (m := session.montage()) else [])
        for i, img in enumerate(images):
            with open(path.replace(".json", f"_img{i}.png"), "wb") as f:
                f.write(img)
    schema = FinalReport if final else Step
    for attempt in range(2):
        ans = ask_qwen(session.a, session.prompt(final), images, schema, path.replace(".json", f"_try{attempt}.json"))
        if ans:
            return ans
    return None


def check_cites(cites, session):
    known = {p["id"] for p in session.points} | {e["id"] for e in session.experiments} | {m["id"] for m in session.moose}
    return sorted({c for c in cites if c.strip() not in known})


def write_report(session, final):
    a, s = session.a, session
    b = boundary(s.points, s.space)
    L = [f"# Exploration: {s.goal['description']}", "",
         f"Run: {a.run}; base case: {a.base_case}; axes: {', '.join(s.space.describe())}",
         f"Points evaluated: {len(s.points)} / {a.budget}; {a.samples} rollouts of {s.n} frames each; "
         f"model: {a.qwen_model}", "",
         "## Fitted boundary (code)", "", f"{(b or {}).get('text') or (b or {}).get('note') or 'not enough points'}",
         f"Classification accuracy of the fit: {(b or {}).get('accuracy')}", "", "![phase](phase.png)", "",
         "## Expression tests (code)", ""]
    tests = [e["result"] for e in s.experiments if e["result"]]
    L += [f"- `{t['expr']}`: AUC {t['auc_expr']}, single parameters "
          f"{ {k[4:]: v for k, v in t.items() if k.startswith('auc_') and k != 'auc_expr'} }, "
          f"line consistency {t.get('line_consistency')}, beats single parameters: {t.get('beats_single_parameters')}"
          for t in tests] or ["(none)"]
    if final:
        bad = check_cites(final.cites + [c for x in final.concepts for c in x.evidence], s)
        L += ["", f"## Qwen conclusion (draft, verify with MOOSE)", ""]
        if bad:
            L += [f"Unknown ids cited: {', '.join(bad)}", ""]
        L += [final.summary, "", f"Region: {final.region}", "", f"Boundary: {final.boundary}", "", "Concepts:"]
        L += [f"- `{c.expr}`: {c.verdict} [{', '.join(c.evidence)}]" for c in final.concepts] or ["- (none)"]
        L += ["", "Caveats:"] + [f"- {c}" for c in final.caveats]
    L += ["", "## MOOSE candidates", "", "Run these before reporting a result (moose_candidates.json):", ""]
    L += [f"- {m['id']} ({m['experiment']}): {json.dumps({k: m['params'][k] for k in s.space.axes})}" for m in s.moose] \
        or ["(none)"]
    with open(os.path.join(a.output, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(a.output, "phase.png"), "wb") as f:
        f.write(s.phase_plot())


def main(argv=None):
    a = parse_args(argv)
    if not a.output:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        a.output = os.path.join(a.run, "explore", f"{stamp}_{'_'.join(a.axes)}")
    os.makedirs(a.output, exist_ok=True)
    s = Session(a)
    if a.resume:
        s.load()
    print(f"goal: {s.goal['description']}\n" + "\n".join(s.space.describe()) +
          f"\neach point: {a.samples} rollouts x {s.n} frames x {a.steps} DDIM steps; budget {a.budget} points")

    finished = False
    try:
        while s.iteration < a.max_iters and s.budget_left() > 0:
            step = ask(s, False, os.path.join(a.output, f"iter_{s.iteration + 1:02d}", "step.json"))
            if step is None:
                print("[WARN] no valid answer from Qwen; falling back to refine")
                step = Step(observation="fallback", hypothesis="", action=Action(tool="refine", k=6),
                            conclusion_so_far="", confidence=1, cites=[])
            print(f"iter {s.iteration + 1}: {step.action.tool} | {step.hypothesis[:100]}")
            if step.action.tool == "finish":
                finished = True
                s.iteration += 1
                break
            rec = s.execute(step.action)
            rec["qwen"] = step.model_dump()
            rec["unknown_cites"] = check_cites(step.cites, s)
            print(f"  {rec['id']}: {rec.get('summary', rec['new'])} {rec['result'] or ''} {rec['notes'] or ''}")
            s.iteration += 1
            s.save()
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        print(f"[WARN] Ollama request failed ({e}); the session is saved, continue with --resume {a.output}")
        s.save()
        write_report(s, None)
        return
    except KeyboardInterrupt:
        print(f"interrupted; continue with --resume {a.output}")
        s.save()
        write_report(s, None)
        return

    reason = "Qwen finished" if finished else ("budget used" if s.budget_left() <= 0 else "max iterations reached")
    print(f"stopping: {reason}; writing the final report")
    final = None
    try:
        final = ask(s, True, os.path.join(a.output, "final", "report.json"))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        print(f"[WARN] final report request failed: {e}")
    if final:
        for pt in final.moose_points:
            s.propose_moose(Action(tool="propose_moose", points=[pt]), "final", [])
        save_json(final.model_dump(), os.path.join(a.output, "final_report.json"))
    s.save()
    write_report(s, final)
    torch.cuda.empty_cache()
    print(f"results in {a.output}")


if __name__ == "__main__":
    main()
