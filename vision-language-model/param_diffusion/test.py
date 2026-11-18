"""Generate images for given parameters and times; compare with the simulated JPGs when available.

# parameters and times of a simulated case (all its frame times, or --times); metrics vs its JPGs
python test.py --run runs/base --case ../data/case_001
python test.py --run runs/base --case ../data/case_001 --times 10 50 95.323235 --num-samples 4

# new parameters (JSON with the same keys as params.json), no reference
python test.py --run runs/base --params my_params.json --times 0 20 40 60 80 100

Outputs in --output (default <run>/test/<case or params file name>):
  images/<var>/<var>_t<time>_s<k>.png   generated images (lossless PNG, model resolution)
  compare_t<time>.png    generated (and simulated, |difference|) images of each variable
  evolution.png          eta (or the first variable) over the requested times
  metrics.csv            per (time, sample) metrics vs the simulated JPGs;  metrics.json  means
  metrics_vs_time.png
Times without a simulated JPG (relative tolerance --time-tol) have no reference.
Times outside the training range [0, t_max] are extrapolation and are flagged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image

from data import CondNormalizer, decode, load_case
from metrics import field_metrics, metric_names
from model import ParamDiffusion, sample
from plots import plot_comparison, plot_evolution


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="training output folder")
    p.add_argument("--model", default="best_model", help="best_model, final_model, or a model folder")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--case", help="case folder: parameters from params.json, reference JPGs from image_files")
    src.add_argument("--params", help="JSON file with simulation parameters")
    p.add_argument("--times", type=float, nargs="+", help="times (default with --case: every frame)")
    p.add_argument("--every", type=int, default=1, help="with --case and no --times: every n-th frame")
    p.add_argument("--num-samples", type=int, default=1, help="samples per time (spread = model uncertainty)")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance scale (1 = off)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-tol", type=float, default=1e-6)
    p.add_argument("--output")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = args.model if os.path.isdir(args.model) else os.path.join(args.run, args.model)
    model = ParamDiffusion.from_pretrained(model_dir).to(device).eval()
    ddim = DDIMScheduler.from_pretrained(os.path.join(args.run, "scheduler"))
    with open(os.path.join(args.run, "normalizers.json")) as f:
        d = json.load(f)
    variables, cond_norm = d["variables"], CondNormalizer.from_dict(d["cond"])
    size = model.config.sample_size

    ref_frames, ref_times = None, None
    if args.case:
        ref_frames, ref_times, params = load_case(args.case, size, variables)
        name = os.path.basename(os.path.normpath(args.case))
        times = args.times if args.times else ref_times[::args.every].tolist()
        with open(os.path.join(args.run, "split.json")) as f:
            split = json.load(f)
        role = "train" if name in split["train"] else "val" if name in split["val"] else "unseen"
        print(f"{name}: {role} case for this run")
    else:
        if not args.times:
            raise SystemExit("--params needs --times")
        with open(args.params) as f:
            params = json.load(f)
        name, times = os.path.splitext(os.path.basename(args.params))[0], args.times
    extra = sorted(set(params) - set(cond_norm.names))
    if extra:
        print(f"ignoring parameters the model was not trained on: {extra}")
    outside = [t for t in times if t < 0 or t > cond_norm.t_max]
    if outside:
        print(f"warning: times {outside} are outside the training range [0, {cond_norm.t_max:g}] (extrapolation)")

    out_dir = args.output or os.path.join(args.run, "test", name)
    for v in variables:
        os.makedirs(os.path.join(out_dir, "images", v), exist_ok=True)

    # One condition per (time, sample).
    conds = [(t, k) for t in times for k in range(args.num_samples)]
    p = cond_norm.params(params)
    g = torch.Generator(device=device).manual_seed(args.seed)
    gen = []
    for s in range(0, len(conds), args.batch_size):
        chunk = conds[s:s + args.batch_size]
        x = sample(model, ddim, p.expand(len(chunk), -1), cond_norm.time([t for t, _ in chunk]),
                   args.steps, args.guidance, generator=g)
        gen.append(decode(x.float().cpu()))
        print(f"generated {min(s + args.batch_size, len(conds))}/{len(conds)}", flush=True)
    gen = torch.cat(gen)
    for (t, k), x in zip(conds, gen):
        for j, v in enumerate(variables):
            img = (x[3 * j:3 * j + 3].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(img).save(os.path.join(out_dir, "images", v, f"{v}_t{t:.6f}_s{k}.png"))

    def reference(t):
        if ref_times is None:
            return None
        j = int(np.argmin(np.abs(ref_times - t)))
        return ref_frames[j] if abs(ref_times[j] - t) <= args.time_tol * max(1.0, abs(t)) else None

    names = metric_names(variables)
    k_evo = variables.index("eta") if "eta" in variables else 0
    rows, evo_gen, evo_ref = [], [], []
    for t in times:
        sel = [i for i, (tc, _) in enumerate(conds) if tc == t]
        g_t, r = gen[sel], reference(t)
        r_t = None if r is None else r.expand_as(g_t)
        plot_comparison(os.path.join(out_dir, f"compare_t{t:.6f}.png"), g_t.numpy(), variables,
                        None if r_t is None else r_t.numpy(), [f"t={t:g} s{k}" for k in range(len(sel))])
        evo_gen.append(g_t[0].numpy())
        evo_ref.append(None if r is None else r.numpy())
        if r_t is not None:
            m = field_metrics(g_t, r_t, variables)
            for k in range(len(sel)):
                rows.append({"time": t, "sample": k, **{n: m[n][k].item() for n in names}})
    plot_evolution(os.path.join(out_dir, "evolution.png"), times, evo_gen,
                   evo_ref if any(e is not None for e in evo_ref) else None, k_evo, variables[k_evo])

    if rows:
        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, ["time", "sample"] + names)
            w.writeheader(); w.writerows(rows)
        summary = {n: float(np.mean([r[n] for r in rows])) for n in names}
        summary["num_compared"] = len(rows)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("mean vs simulation: " + "  ".join(f"{k} {v:.4g}" for k, v in summary.items()))
        _plot_metrics_vs_time(os.path.join(out_dir, "metrics_vs_time.png"), rows, names)
    print(f"outputs in {out_dir}")


def _plot_metrics_vs_time(path, rows, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3), squeeze=False)
    t = np.array([r["time"] for r in rows])
    for ax, n in zip(axes[0], names):
        ax.plot(t, [r[n] for r in rows], "o", ms=3)
        ax.set_title(n, fontsize=9); ax.set_xlabel("time"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
