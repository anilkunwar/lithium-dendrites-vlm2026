"""Frame sequences from step0 output: data/<case>/npy_files/<time>.npy, shape (H, W, C)."""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CHANNELS = ("eta", "c", "pot")


def discover_cases(data_root, names=None):
    dirs = sorted(os.path.dirname(d) for d in glob.glob(os.path.join(data_root, "*", "npy_files")))
    cases = {os.path.basename(d): d for d in dirs if glob.glob(os.path.join(d, "npy_files", "*.npy"))}
    if names:
        missing = [n for n in names if n not in cases]
        if missing:
            raise FileNotFoundError(f"cases not found under {data_root}: {missing}")
        cases = {n: cases[n] for n in names}
    if not cases:
        raise FileNotFoundError(f"no data/<case>/npy_files/*.npy under {data_root}")
    return cases


def load_case(case_dir, resolution=None):
    """Return frames (T, C, H, W) float32 and times (T,) sorted by time."""
    files = glob.glob(os.path.join(case_dir, "npy_files", "*.npy"))
    files.sort(key=lambda f: float(os.path.basename(f)[:-4]))
    times = np.array([float(os.path.basename(f)[:-4]) for f in files])
    arrays = [np.load(f) for f in files]
    bad = [os.path.basename(f) for f, a in zip(files, arrays)
           if a.ndim != 3 or a.shape[-1] != len(CHANNELS) or a.shape != arrays[0].shape]
    if bad:
        raise ValueError(f"{case_dir}: frames must all be (H, W, {len(CHANNELS)}); bad files: {bad[:5]}")
    frames = np.stack(arrays).astype(np.float32)  # (T, H, W, C)
    # griddata leaves NaN outside the convex hull; replace with per-frame channel means.
    if np.isnan(frames).any():
        means = np.nanmean(frames, axis=(1, 2), keepdims=True)
        frames = np.where(np.isnan(frames), means, frames)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    if resolution and frames.shape[-1] != resolution:
        frames = F.interpolate(frames, size=(resolution, resolution), mode="bilinear",
                               antialias=True, align_corners=False)
    return frames, times


class Normalizer:
    """Per-channel min/max to [-1, 1]; statistics come from training frames only."""

    def __init__(self, lo, hi):
        self.lo = torch.as_tensor(lo, dtype=torch.float32).view(-1, 1, 1)
        self.hi = torch.as_tensor(hi, dtype=torch.float32).view(-1, 1, 1)

    @classmethod
    def fit(cls, frame_list):
        x = torch.cat(frame_list)
        return cls(x.amin(dim=(0, 2, 3)), x.amax(dim=(0, 2, 3)))

    def encode(self, x):
        lo, hi = self.lo.to(x), self.hi.to(x)
        return 2 * (x - lo) / (hi - lo).clamp_min(1e-8) - 1

    def decode(self, x):
        lo, hi = self.lo.to(x), self.hi.to(x)
        return (x + 1) / 2 * (hi - lo) + lo

    def to_dict(self):
        return {"lo": self.lo.flatten().tolist(), "hi": self.hi.flatten().tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["lo"], d["hi"])


def log_gaps(times):
    """Time gaps between consecutive frames, log-scaled because MOOSE uses adaptive steps."""
    return torch.log1p(torch.as_tensor(np.diff(times), dtype=torch.float32))


def split(cases, num_context, val_cases=None, train_frac=0.8):
    """Return {case: (first_target, last_target)} ranges for train and test.

    Held-out cases (val_cases) are tested from frame num_context onward.
    Otherwise each case is split in time: early targets for training, later ones for testing.
    """
    train, test = {}, {}
    for name, (frames, _) in cases.items():
        T = len(frames)
        if T <= num_context:
            print(f"[WARN] skipping {name}: {T} frames, need more than --context {num_context}")
            continue
        if val_cases:
            (test if name in val_cases else train)[name] = (num_context, T - 1)
        else:
            cut = min(max(num_context + 1, int(T * train_frac)), T)
            train[name] = (num_context, cut - 1)
            test[name] = (cut, T - 1)
    return train, test


class FrameWindowDataset(Dataset):
    """Samples (K context frames -> next frame) windows with their time gaps."""

    def __init__(self, cases, ranges, normalizer, num_context, flip_y=True):
        self.cases = {n: (normalizer.encode(f), t) for n, (f, t) in cases.items() if n in ranges}
        self.K = num_context
        self.flip_y = flip_y
        self.index = [(n, i) for n, (a, b) in ranges.items() for i in range(a, b + 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        name, i = self.index[idx]
        frames, times = self.cases[name]
        window = frames[i - self.K:i + 1]  # (K+1, C, H, W)
        # y-mirror symmetry of the domain (axis 2 is y in step0's grid).
        if self.flip_y and torch.rand(()) < 0.5:
            window = window.flip(-2)
        return {
            "context": window[:-1].flatten(0, 1),
            "target": window[-1],
            "gaps": log_gaps(times[i - self.K:i + 1]),
        }


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
