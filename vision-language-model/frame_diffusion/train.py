#!/usr/bin/env python3
"""Train a conditional diffusion model that predicts the next frame from K previous frames.

python frame_diffusion/train.py --data data --output runs/fd_k4
accelerate launch frame_diffusion/train.py ...   # multi-GPU
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_constant_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from data import CHANNELS, FrameWindowDataset, Normalizer, discover_cases, load_case, save_json, split
from model import FramePredictor, sample_next

VAL_WINDOWS = 256  # fixed, evenly spaced held-out windows used for val_mse
PATIENCE = 10  # stop after this many validations without a new best val_mse


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--cases", nargs="*", help="case names to use (default: all converted cases)")
    p.add_argument("--val-cases", nargs="*", help="held-out cases; default splits each case in time")
    p.add_argument("--train-frac", type=float, default=0.8, help="time split when --val-cases is empty")
    p.add_argument("--output", default="runs/frame_diffusion")
    p.add_argument("--context", type=int, default=4, help="number of conditioning frames K")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--block-out-channels", type=int, nargs="+", default=[64, 128, 256, 256, 512])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=200000,
                   help="upper bound; training stops earlier once val_mse stops improving")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--prediction-type", default="v_prediction", choices=["epsilon", "v_prediction", "sample"])
    p.add_argument("--train-timesteps", type=int, default=1000)
    p.add_argument("--context-noise", type=float, default=0.02,
                   help="max std of Gaussian noise on context frames; reduces rollout drift")
    p.add_argument("--no-flip", action="store_true", help="disable y-mirror augmentation")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-steps", type=int, default=25, help="DDIM steps for validation")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--resume", help="accelerate state directory to resume from")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def save_model(accelerator, model, ema, out_dir):
    unwrapped = accelerator.unwrap_model(model)
    ema.store(unwrapped.parameters())
    ema.copy_to(unwrapped.parameters())
    unwrapped.save_pretrained(out_dir)
    ema.restore(unwrapped.parameters())


@torch.no_grad()
def validate(model, ema, scheduler, loader, steps, device):
    """One-step-ahead MSE (normalized units) with EMA weights on held-out windows."""
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    model.eval()
    g = torch.Generator(device=device).manual_seed(0)
    se, n = 0.0, 0
    for batch in loader:
        ctx, tgt, gaps = (batch[k].to(device) for k in ("context", "target", "gaps"))
        pred = sample_next(model, scheduler, ctx, gaps, num_steps=steps, generator=g)
        se += F.mse_loss(pred, tgt, reduction="sum").item()
        n += tgt.numel()
    ema.restore(model.parameters())
    model.train()
    return se / max(n, 1)


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision,
                              gradient_accumulation_steps=args.grad_accum)
    set_seed(args.seed)

    cases = {n: load_case(d, args.resolution) for n, d in discover_cases(args.data, args.cases).items()}
    train_ranges, test_ranges = split(cases, args.context, args.val_cases, args.train_frac)
    train_ranges = {n: r for n, r in train_ranges.items() if r[0] <= r[1]}
    if not train_ranges:
        raise ValueError("no training windows; reduce --context or add frames")
    # Statistics from frames seen in training only.
    normalizer = Normalizer.fit([cases[n][0][:b + 1] for n, (_, b) in train_ranges.items()])

    train_ds = FrameWindowDataset(cases, train_ranges, normalizer, args.context, flip_y=not args.no_flip)
    test_ranges = {n: r for n, r in test_ranges.items() if r[0] <= r[1]}
    val_ds = FrameWindowDataset(cases, test_ranges, normalizer, args.context, flip_y=False) if test_ranges else None
    accelerator.print(f"cases={list(cases)} train windows={len(train_ds)} "
                      f"test windows={len(val_ds) if val_ds else 0}")

    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)
        save_json({
            "args": vars(args),
            "channels": list(CHANNELS),
            "normalizer": normalizer.to_dict(),
            "train_ranges": train_ranges,
            "test_ranges": test_ranges,
        }, os.path.join(args.output, "run_config.json"))

    model = FramePredictor(
        num_channels=len(CHANNELS), num_context=args.context, sample_size=args.resolution,
        block_out_channels=tuple(args.block_out_channels),
        attention_levels=tuple(i >= len(args.block_out_channels) - 2 for i in range(len(args.block_out_channels))),
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.train_timesteps, beta_schedule="squaredcos_cap_v2",
                                    prediction_type=args.prediction_type, clip_sample=False)
    sample_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    if accelerator.is_main_process:
        noise_scheduler.save_pretrained(os.path.join(args.output, "scheduler"))
    ema = EMAModel(model.parameters(), decay=args.ema_decay)

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) >= args.batch_size,
                        num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    val_loader = None
    if val_ds:
        idx = torch.linspace(0, len(val_ds) - 1, min(VAL_WINDOWS, len(val_ds))).long().unique().tolist()
        val_loader = DataLoader(Subset(val_ds, idx), batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    lr_scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_steps)
    model, optimizer, loader, lr_scheduler = accelerator.prepare(model, optimizer, loader, lr_scheduler)

    step, best = 0, {"val_mse": float("inf"), "step": 0, "bad": 0}
    if args.resume:
        accelerator.load_state(args.resume)
        ema.load_state_dict(torch.load(os.path.join(args.resume, "ema.pt"), map_location="cpu"))
        step = int(args.resume.rstrip("/").split("-")[-1])
        if os.path.exists(os.path.join(args.resume, "best.json")):
            with open(os.path.join(args.resume, "best.json")) as f:
                best = json.load(f)
    ema.to(accelerator.device)
    model_dir = os.path.join(args.output, "model")

    epochs = math.ceil((args.max_steps - step) * args.grad_accum / max(len(loader), 1)) + 1
    stop = False
    progress = tqdm(total=args.max_steps, initial=step, disable=not accelerator.is_local_main_process)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            ctx, tgt, gaps = batch["context"], batch["target"], batch["gaps"]
            if args.context_noise > 0:
                sigma = torch.rand(ctx.shape[0], 1, 1, 1, device=ctx.device) * args.context_noise
                ctx = ctx + sigma * torch.randn_like(ctx)
            noise = torch.randn_like(tgt)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (tgt.shape[0],), device=tgt.device)
            noisy = noise_scheduler.add_noise(tgt, noise, t)
            if args.prediction_type == "epsilon":
                target = noise
            elif args.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(tgt, noise, t)
            else:
                target = tgt

            with accelerator.accumulate(model):
                pred = model(noisy, t, ctx, gaps)
                loss = F.mse_loss(pred.float(), target.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue
            ema.step(model.parameters())
            step += 1
            progress.update(1)
            logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}

            if val_loader and accelerator.is_main_process and step % args.val_every == 0:
                logs["val_mse"] = validate(accelerator.unwrap_model(model), ema, sample_scheduler,
                                           val_loader, args.val_steps, accelerator.device)
                if logs["val_mse"] < best["val_mse"]:
                    best = {"val_mse": logs["val_mse"], "step": step, "bad": 0}
                    save_model(accelerator, model, ema, model_dir)
                else:
                    best["bad"] += 1
                accelerator.print(f"step {step}: val_mse {logs['val_mse']:.5f} "
                                  f"(best {best['val_mse']:.5f} at step {best['step']})")
                if best["bad"] >= PATIENCE:
                    accelerator.print(f"val_mse has not improved for {PATIENCE} validations; stopping")
                    accelerator.set_trigger()
            progress.set_postfix(loss=f"{logs['loss']:.4f}")
            if accelerator.is_main_process and (step % 50 == 0 or "val_mse" in logs):
                with open(os.path.join(args.output, "train_log.jsonl"), "a") as f:
                    f.write(json.dumps({"step": step, **logs}) + "\n")

            stop = accelerator.check_trigger() or step >= args.max_steps
            if step % args.save_every == 0 or stop:
                accelerator.wait_for_everyone()
                state_dir = os.path.join(args.output, f"checkpoint-{step}")
                accelerator.save_state(state_dir)
                if accelerator.is_main_process:
                    torch.save(ema.state_dict(), os.path.join(state_dir, "ema.pt"))
                    save_json(best, os.path.join(state_dir, "best.json"))
                    if not val_loader:  # nothing to select on: keep the latest weights
                        save_model(accelerator, model, ema, model_dir)
            if stop:
                break
        if stop:
            break

    if val_loader:
        accelerator.print(f"best EMA model (step {best['step']}, val_mse {best['val_mse']:.5f}) in {model_dir}")
    else:
        accelerator.print(f"EMA model saved to {model_dir}")


if __name__ == "__main__":
    main()
