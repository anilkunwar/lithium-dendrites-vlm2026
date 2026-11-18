"""Train the (parameters, time) -> image diffusion model on the step0 JPGs.

python train.py --data ../data --output runs/base
python train.py --data ../data --output runs/base --resume          # continue from runs/base/checkpoints/last.pt
python train.py --data ../data --output runs/s256 --size 256 --batch-size 8 --block-out-channels 64 128 256 256 512 \
    --attention-levels 0 0 0 1 1

Outputs in --output:
  config.json, normalizers.json, split.json   run setup (test.py reads these)
  metrics_train.csv    step, loss, lr, grad_norm (means over each --log-every interval)
  metrics_eval.csv     every --eval-every steps, EMA weights:
                         val_loss / train_loss_fixed  loss with fixed timesteps and noise (comparable over steps)
                         rmse_<var>, psnr_<var>             DDIM samples vs simulated JPGs (RGB in [0, 1])
                         eta_iou, front_err                  deposit shape, eta decoded from the jet colors
  training_curves.png  redrawn after each evaluation
  samples/step_*.png   generated vs simulated val images
  best_model/          EMA weights with the lowest val_loss;  final_model/  EMA weights at the end
  checkpoints/last.pt  model, EMA, optimizer, scheduler, step (for --resume)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

from data import VARIABLES, CondNormalizer, FieldDataset, decode, discover_cases, load_case, select, split
from metrics import field_metrics, metric_names
from model import ParamDiffusion, sample
from plots import plot_comparison, plot_training


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--data", default="../data", help="folder with <case>/image_files/<var>/*.jpg and <case>/params.json")
    p.add_argument("--variables", nargs="+", default=list(VARIABLES), choices=VARIABLES,
                   help="image_files/<var> folders to learn; each adds 3 RGB channels")
    p.add_argument("--cases", nargs="*", help="use only these cases (default: all)")
    p.add_argument("--val-cases", nargs="*", help="held-out cases (default: random --val-frac of the cases)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-every", type=int, default=5,
                   help="with a single case: hold out every n-th frame instead of whole cases")
    p.add_argument("--size", type=int, default=128, help="training resolution (JPGs are resized)")
    p.add_argument("--log-params", nargs="*", default=["ko", "Noise"],
                   help="parameters z-scored in log10 (sampled over decades)")
    p.add_argument("--flip", action="store_true", help="random y-flip (only if top/bottom BCs are the same)")
    # model
    p.add_argument("--block-out-channels", type=int, nargs="+", default=[64, 128, 256, 256])
    p.add_argument("--attention-levels", type=int, nargs="+", default=[0, 0, 1, 1])
    p.add_argument("--layers-per-block", type=int, default=2)
    p.add_argument("--time-freqs", type=int, default=8)
    p.add_argument("--cond-dropout", type=float, default=0.1,
                   help="probability of training with the null condition (enables guidance in test.py)")
    p.add_argument("--num-train-timesteps", type=int, default=1000)
    p.add_argument("--beta-schedule", default="squaredcos_cap_v2")
    p.add_argument("--prediction-type", choices=["epsilon", "v_prediction"], default="v_prediction")
    # optimization
    p.add_argument("--output", default="runs/base")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9995)
    p.add_argument("--mixed-precision", choices=["no", "bf16", "fp16"], default="bf16")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    # logging / evaluation
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--val-loss-samples", type=int, default=256, help="max samples for the fixed-noise losses")
    p.add_argument("--val-loss-timesteps", type=int, default=10, help="diffusion timesteps per sample in val_loss")
    p.add_argument("--eval-samples", type=int, default=8, help="val samples generated for image metrics")
    p.add_argument("--sample-steps", type=int, default=50, help="DDIM steps during evaluation")
    return p.parse_args()


def build_datasets(args, cond_norm=None):
    """cond_norm: saved normalizer to reuse (on --resume) instead of fitting a new one."""
    cases = discover_cases(args.data, args.cases, args.variables)
    train_sel, val_sel = split(cases, args.val_cases, args.val_frac, args.val_every, args.seed)
    print(f"{len(cases)} cases, variables {args.variables}, resolution {args.size}; reading JPGs ...", flush=True)
    loaded = {n: load_case(d, args.size, args.variables) for n, d in cases.items()}

    def items(sel):
        out = []
        for n, s in sel.items():
            frames, times, params = loaded[n]
            out.append((n, frames, times, params, select(s, len(times))))
        return out

    train_items, val_items = items(train_sel), items(val_sel)
    names = list(train_items[0][3])
    for n, _, _, p, _ in train_items + val_items:
        if set(p) != set(names):
            raise ValueError(f"{n}/params.json has keys {sorted(p)}, expected {sorted(names)}")

    if cond_norm is None:
        train_times = np.concatenate([t[i] for _, _, t, _, i in train_items])
        cond_norm = CondNormalizer.fit([p for _, _, _, p, _ in train_items], train_times, args.log_params)
    train_ds = FieldDataset(train_items, cond_norm, flip=args.flip)
    val_ds = FieldDataset(val_items, cond_norm)
    split_info = {
        "train": {n: [float(t[j]) for j in i] for n, _, t, _, i in train_items},
        "val": {n: [float(t[j]) for j in i] for n, _, t, _, i in val_items},
    }
    if len(cases) == 1:
        print("only one case: parameters are constant, the model learns the time dependence only; "
              "validation holds out frames of the same case")
    print(f"train {len(train_ds)} samples, val {len(val_ds)} samples")
    return train_ds, val_ds, cond_norm, split_info


def diffusion_target(scheduler, x0, noise, t):
    if scheduler.config.prediction_type == "epsilon":
        return noise
    return scheduler.get_velocity(x0, noise, t)


@torch.no_grad()
def fixed_noise_loss(model, scheduler, ds, args, device, amp_dtype):
    """Mean diffusion loss with the same timesteps and noise at every evaluation."""
    n = min(len(ds), args.val_loss_samples)
    idx = np.linspace(0, len(ds) - 1, n).round().astype(int)
    ts = torch.linspace(0, scheduler.config.num_train_timesteps - 1, args.val_loss_timesteps).round().long()
    g = torch.Generator(device="cpu").manual_seed(1234)
    total, count = 0.0, 0
    for start in range(0, n, args.batch_size):
        b = idx[start:start + args.batch_size]
        x0 = torch.stack([ds.fields[i] for i in b]).to(device)
        params = torch.stack([ds.params[i] for i in b]).to(device)
        tm = torch.stack([ds.times[i] for i in b]).to(device)
        for t in ts:
            noise = torch.randn(x0.shape, generator=g).to(device)
            tt = t.expand(len(b)).to(device)
            xt = scheduler.add_noise(x0, noise, tt)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(xt, tt, params, tm)
            total += F.mse_loss(out.float(), diffusion_target(scheduler, x0, noise, tt)).item() * len(b)
            count += len(b)
    return total / count if count else float("nan")


@torch.no_grad()
def sample_metrics(model, ddim, ds, args, device, step, out_dir):
    n = min(len(ds), args.eval_samples)
    if n == 0:
        return {k: float("nan") for k in metric_names(args.variables)}
    idx = np.linspace(0, len(ds) - 1, n).round().astype(int)
    ref = decode(torch.stack([ds.fields[i] for i in idx]))
    params = torch.stack([ds.params[i] for i in idx])
    tm = torch.stack([ds.times[i] for i in idx])
    g = torch.Generator(device=device).manual_seed(4321)
    gen = []
    for s in range(0, n, args.batch_size):
        x = sample(model, ddim, params[s:s + args.batch_size], tm[s:s + args.batch_size],
                   args.sample_steps, generator=g)
        gen.append(decode(x.float().cpu()))
    gen = torch.cat(gen)
    m = {k: v.mean().item() for k, v in field_metrics(gen, ref, args.variables).items()}
    show = min(n, 4)
    titles = [f"{ds.cases[i]} t={ds.raw_times[i]:g}" for i in idx[:show]]
    plot_comparison(os.path.join(out_dir, "samples", f"step_{step:07d}.png"),
                    gen[:show].numpy(), args.variables, ref[:show].numpy(), titles)
    return m


class CSVLog:
    def __init__(self, path, fields, resume):
        self.path, self.fields = path, fields
        if not (resume and os.path.isfile(path)):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(fields)

    def truncate_after(self, step):
        """On resume, drop rows logged after the checkpoint so the curves have no duplicates."""
        with open(self.path) as f:
            rows = list(csv.DictReader(f))
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, self.fields)
            w.writeheader()
            w.writerows(r for r in rows if int(r["step"]) <= step)

    def write(self, row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.fields).writerow(row)


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p.detach(), 1 - decay)
    for be, b in zip(ema.buffers(), model.buffers()):
        be.copy_(b)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"no": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.mixed_precision]
    if device.type != "cuda":
        amp_dtype = None
    os.makedirs(os.path.join(args.output, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "checkpoints"), exist_ok=True)

    if args.resume:
        # Same normalization as the interrupted run, so the model sees the same inputs.
        with open(os.path.join(args.output, "normalizers.json")) as f:
            d = json.load(f)
        if d["variables"] != args.variables:
            raise ValueError(f"--variables {args.variables} differ from the run's {d['variables']}")
        train_ds, val_ds, cond_norm, split_info = build_datasets(args, CondNormalizer.from_dict(d["cond"]))
    else:
        train_ds, val_ds, cond_norm, split_info = build_datasets(args)
        with open(os.path.join(args.output, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        with open(os.path.join(args.output, "normalizers.json"), "w") as f:
            json.dump({"variables": args.variables, "cond": cond_norm.to_dict()}, f, indent=2)
        with open(os.path.join(args.output, "split.json"), "w") as f:
            json.dump(split_info, f, indent=2)

    if len(args.attention_levels) != len(args.block_out_channels):
        raise ValueError("--attention-levels needs one entry per --block-out-channels")
    model = ParamDiffusion(
        num_channels=3 * len(args.variables), num_params=len(cond_norm.names), sample_size=args.size,
        block_out_channels=tuple(args.block_out_channels), layers_per_block=args.layers_per_block,
        attention_levels=tuple(bool(a) for a in args.attention_levels), time_freqs=args.time_freqs,
        cond_dropout=args.cond_dropout,
    ).to(device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

    noise_sched = DDPMScheduler(num_train_timesteps=args.num_train_timesteps, beta_schedule=args.beta_schedule,
                                prediction_type=args.prediction_type, clip_sample=True)
    ddim = DDIMScheduler.from_config(noise_sched.config)
    noise_sched.save_pretrained(os.path.join(args.output, "scheduler"))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    step, best = 0, math.inf
    ckpt_path = os.path.join(args.output, "checkpoints", "last.pt")
    train_fields = ["step", "loss", "lr", "grad_norm", "samples_per_s"]
    eval_fields = ["step", "val_loss", "train_loss_fixed"] + metric_names(args.variables)
    train_log = CSVLog(os.path.join(args.output, "metrics_train.csv"), train_fields, args.resume)
    eval_log = CSVLog(os.path.join(args.output, "metrics_eval.csv"), eval_fields, args.resume)
    if args.resume:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); lr_sched.load_state_dict(ck["lr_sched"])
        scaler.load_state_dict(ck["scaler"])
        step, best = ck["step"], ck["best"]
        train_log.truncate_after(step); eval_log.truncate_after(step)
        print(f"resumed at step {step}")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) >= args.batch_size,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)

    def batches():
        while True:
            yield from loader

    it = batches()
    acc = {"loss": 0.0, "grad_norm": 0.0, "n": 0}
    t0 = time.time()
    model.train()
    while step < args.steps:
        b = next(it)
        x0, params, tm = b["field"].to(device), b["params"].to(device), b["time"].to(device)
        B = x0.shape[0]
        noise = torch.randn_like(x0)
        t = torch.randint(0, args.num_train_timesteps, (B,), device=device)
        xt = noise_sched.add_noise(x0, noise, t)
        drop = torch.rand(B, device=device) < args.cond_dropout if args.cond_dropout > 0 else None
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(xt, t, params, tm, drop)
        loss = F.mse_loss(out.float(), diffusion_target(noise_sched, x0, noise, t))

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        lr_sched.step()
        step += 1
        # EMA warm-up: average over roughly the first steps instead of the random init.
        ema_update(ema, model, min(args.ema_decay, (1 + step) / (10 + step)))

        acc["loss"] += loss.item(); acc["grad_norm"] += grad_norm.item(); acc["n"] += 1
        if step % args.log_every == 0:
            dt = time.time() - t0
            row = {"step": step, "loss": acc["loss"] / acc["n"], "lr": lr_sched.get_last_lr()[0],
                   "grad_norm": acc["grad_norm"] / acc["n"], "samples_per_s": acc["n"] * args.batch_size / dt}
            train_log.write(row)
            print(f"step {step:7d}  loss {row['loss']:.5f}  lr {row['lr']:.2e}  "
                  f"grad {row['grad_norm']:.3f}  {row['samples_per_s']:.1f} samples/s", flush=True)
            acc = {"loss": 0.0, "grad_norm": 0.0, "n": 0}
            t0 = time.time()

        if step % args.eval_every == 0 or step == args.steps:
            row = {"step": step,
                   "val_loss": fixed_noise_loss(ema, noise_sched, val_ds, args, device, amp_dtype),
                   "train_loss_fixed": fixed_noise_loss(ema, noise_sched, train_ds, args, device, amp_dtype)}
            row.update(sample_metrics(ema, ddim, val_ds, args, device, step, args.output))
            eval_log.write(row)
            print("eval  " + "  ".join(f"{k} {v:.4g}" for k, v in row.items() if k != "step"), flush=True)
            if row["val_loss"] < best:
                best = row["val_loss"]
                ema.save_pretrained(os.path.join(args.output, "best_model"))
            plot_training(args.output)
            t0 = time.time()

        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "lr_sched": lr_sched.state_dict(), "scaler": scaler.state_dict(),
                        "step": step, "best": best}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    ema.save_pretrained(os.path.join(args.output, "final_model"))
    print(f"done; best val_loss {best:.5f}; curves in {plot_training(args.output)}")


if __name__ == "__main__":
    main()
