"""(parameters, time) -> image samples from step0 output.

Each case has data/<case>/image_files/<var>/<var>_t<time>.jpg (jet colormap, RGB) for each
variable, and data/<case>/params.json with the sampled simulation parameters.
Every time with an image for all requested variables is one training sample:
condition = (case parameters, time), target = the variables' RGB images stacked on channels.

step0 draws each JPG with its own color scale (tripcolor without vmin/vmax), so colors are
relative to each frame; the model learns the images, not absolute field values.
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

VARIABLES = ("eta", "c", "pot")


def image_path(case_dir, var, t_str):
    return os.path.join(case_dir, "image_files", var, f"{var}_t{t_str}.jpg")


def frame_times(case_dir, variables=VARIABLES):
    """Sorted (time strings, times) that have a JPG for every variable."""
    common = None
    for var in variables:
        names = glob.glob(os.path.join(case_dir, "image_files", var, f"{var}_t*.jpg"))
        ts = {m.group(1) for m in (re.fullmatch(rf"{re.escape(var)}_t([-+0-9.eE]+)\.jpg", os.path.basename(n))
                                   for n in names) if m}
        common = ts if common is None else common & ts
    t_strs = sorted(common or [], key=float)
    return t_strs, np.array([float(t) for t in t_strs])


def discover_cases(data_root, names=None, variables=VARIABLES):
    """{case name: case dir} for every data/<case> with image_files/<var>/*.jpg and params.json."""
    cases = {}
    for d in sorted(glob.glob(os.path.join(data_root, "*", "image_files"))):
        case_dir = os.path.dirname(d)
        if os.path.isfile(os.path.join(case_dir, "params.json")) and len(frame_times(case_dir, variables)[0]):
            cases[os.path.basename(case_dir)] = case_dir
    if names:
        missing = [n for n in names if n not in cases]
        if missing:
            raise FileNotFoundError(f"cases with image_files/{{{','.join(variables)}}} and params.json "
                                    f"not found under {data_root}: {missing}")
        cases = {n: cases[n] for n in names}
    if not cases:
        raise FileNotFoundError(f"no data/<case>/image_files/<var>/*.jpg with params.json under {data_root}")
    return cases


def read_image(path, size=None):
    """JPG -> (3, size, size) float32 RGB in [0, 1]; row 0 is the top of the image."""
    im = Image.open(path).convert("RGB")
    if size and im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC, reducing_gap=3.0)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)


def load_case(case_dir, size=None, variables=VARIABLES):
    """Images (T, 3 * len(variables), size, size) in [0, 1], times (T,), raw params dict."""
    t_strs, times = frame_times(case_dir, variables)
    frames = torch.stack([torch.cat([read_image(image_path(case_dir, v, t), size) for v in variables])
                          for t in t_strs])
    with open(os.path.join(case_dir, "params.json")) as f:
        params = json.load(f)
    return frames, times, params


def encode(x):
    """[0, 1] images -> [-1, 1] model space."""
    return 2 * x - 1


def decode(x):
    return ((x + 1) / 2).clamp(0, 1)


class CondNormalizer:
    """Parameters -> z-scores (log10 first for `log` names); time -> log1p(t) / log1p(t_max) in [0, 1].

    Parameters constant over the training cases (e.g. a single case) get std 1, i.e. z = 0,
    so the model cannot learn their effect; it only learns the time dependence.
    """

    def __init__(self, names, mean, std, log, t_max):
        self.names, self.log = list(names), list(log)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.t_max = float(t_max)

    def _raw(self, params):
        missing = [n for n in self.names if n not in params]
        if missing:
            raise KeyError(f"parameters missing: {missing}")
        v = np.array([params[n] for n in self.names], dtype=np.float64)
        for i, n in enumerate(self.names):
            if n in self.log:
                if v[i] <= 0:
                    raise ValueError(f"log-scaled parameter {n} must be positive, got {v[i]}")
                v[i] = np.log10(v[i])
        return v

    @classmethod
    def fit(cls, param_dicts, times, log=()):
        names = list(param_dicts[0])
        log = [n for n in log if n in names]
        tmp = cls(names, np.zeros(len(names)), np.ones(len(names)), log, 1.0)
        raw = np.stack([tmp._raw(p) for p in param_dicts])
        std = raw.std(axis=0)
        std[std < 1e-12 * np.maximum(np.abs(raw.mean(axis=0)), 1.0)] = 1.0
        return cls(names, raw.mean(axis=0), std, log, float(np.max(times)))

    def params(self, params):
        return torch.tensor((self._raw(params) - self.mean) / self.std, dtype=torch.float32)

    def time(self, t):
        t = np.asarray(t, dtype=np.float64)
        if (t < 0).any():
            raise ValueError("times must be >= 0")
        return torch.tensor(np.log1p(t) / np.log1p(self.t_max), dtype=torch.float32)

    def to_dict(self):
        return {"names": self.names, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "log": self.log, "t_max": self.t_max}

    @classmethod
    def from_dict(cls, d):
        return cls(d["names"], d["mean"], d["std"], d["log"], d["t_max"])


class FieldDataset(Dataset):
    """Samples in model space: (images (3V,H,W) in [-1,1], params z (P,), time s ()), plus raw time and case."""

    def __init__(self, items, cond_norm, flip=False):
        # items: list of (case, frames (T,3V,H,W) in [0,1], times (T,), params dict, frame indices)
        self.flip = flip
        self.fields, self.params, self.times, self.raw_times, self.cases = [], [], [], [], []
        for case, frames, times, params, idx in items:
            p = cond_norm.params(params)
            for i in idx:
                self.fields.append(encode(frames[i]))
                self.params.append(p)
                self.times.append(cond_norm.time(times[i]))
                self.raw_times.append(float(times[i]))
                self.cases.append(case)

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, i):
        x = self.fields[i]
        # Optional y-flip (image rows): valid only if the top and bottom boundary conditions
        # are the same, since the electrode is the left boundary.
        if self.flip and torch.rand(()) < 0.5:
            x = x.flip(-2)
        return {"field": x, "params": self.params[i], "time": self.times[i]}


def split(cases, val_cases=None, val_frac=0.1, val_every=5, seed=0):
    """Return (train, val) as {case: frame selection} where a selection is 'all' or a held-out rule.

    With >= 2 cases: hold out whole cases (--val-cases, else a random val_frac of them).
    With 1 case: hold out every val_every-th frame (tests interpolation in time only).
    """
    names = list(cases)
    if len(names) == 1:
        return {names[0]: ("every", val_every, False)}, {names[0]: ("every", val_every, True)}
    if not val_cases:
        rng = np.random.default_rng(seed)
        k = max(1, int(round(val_frac * len(names))))
        val_cases = sorted(rng.choice(names, size=k, replace=False).tolist())
    unknown = [n for n in val_cases if n not in cases]
    if unknown:
        raise ValueError(f"unknown --val-cases {unknown}")
    if len(val_cases) == len(names):
        raise ValueError("every case is in the validation set")
    return ({n: "all" for n in names if n not in val_cases}, {n: "all" for n in val_cases})


def select(sel, n_frames):
    if sel == "all":
        return list(range(n_frames))
    _, every, held_out = sel
    # Never hold out the first or last frame, so validation times stay inside the training range.
    held = {i for i in range(1, n_frames - 1) if i % every == 0}
    return sorted(held) if held_out else [i for i in range(n_frames) if i not in held]
