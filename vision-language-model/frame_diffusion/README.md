# Frame diffusion: predicting future phase-field frames

A conditional diffusion model that takes the previous K frames (`eta, c, pot`) and generates
the next frame. Repeating this autoregressively produces a sequence. It is built from
standard components:

- [HuggingFace diffusers](https://github.com/huggingface/diffusers): `UNet2DModel`, `DDPMScheduler`
  for training, `DDIMScheduler` for sampling, and `EMAModel`
- [accelerate](https://github.com/huggingface/accelerate): mixed precision, multiple GPUs, checkpoint resume
- Conditioning: the context frames are concatenated with the noisy target along the channel
  axis (as in MCVD, Voleti et al. 2022, and SR3). MOOSE uses adaptive time steps, so the log time
  gaps between frames are embedded and added to the timestep embedding.

The model is trained from scratch on the physical fields themselves. Pretrained video models
(SVD, CogVideoX) work on RGB natural video in a VAE latent space and accept one conditioning
frame, so they do not fit a 3-channel physical field with multi-frame conditioning.

## Usage

Set up the environment (versions pinned to the tested setup; `requirements.txt` holds the same list for pip):

```bash
conda env create -f frame_diffusion/environment.yml
```

Run from `vision-language-model/`:

```bash
python frame_diffusion/train.py --data data --output runs/fd_k4 --context 4 --max-steps 20000
```

```bash
python frame_diffusion/test.py --run runs/fd_k4
```

Train/test split:

- `--val-cases case_005 case_006`: whole held-out cases (preferred once more cases are converted)
- Otherwise each case is split in time. Windows whose target is in the first `--train-frac`
  (0.8) of the frames are used for training, and the test rolls out over the remaining frames.
  This tests extrapolation to later growth stages.

Useful `test.py` options:

| Option | Purpose |
| --- | --- |
| `--teacher-forcing` | Condition every step on ground-truth frames (one-step error, no accumulation) |
| `--samples 8` | Multiple stochastic rollouts; reports the ensemble mean and standard deviation |
| `--case case_001 --start 10 --num-frames 30` | Choose the starting frame and length |
| `--dt 6.5` | Time step for frames beyond the end of the data |
| `--steps 50` | DDIM sampling steps |

## Outputs

`runs/<name>/`: `model/` (EMA weights, `from_pretrained`), `scheduler/`, `run_config.json`
(arguments, normalization statistics, split), `train_log.jsonl`, and `checkpoint-*` (pass to `--resume`).

`runs/<name>/test/<case>/`:

- `sample_XX/npy_files/<time>.npy`: predicted frames in the same `(H, W, 3)` format as step0
- `metrics.json`: per frame and channel MSE/PSNR/SSIM, plus tip position (fraction of domain width)
  and area fraction of the deposit attached to the left boundary (`eta >= 0.5`, 4-connectivity)
- `comparison.png`: ground truth, prediction, and error; `metrics.png`: metric curves over time

## Notes

- Currently only `case_001` (56 frames, 40 training windows) is converted. That is enough to check
  the pipeline but not to generalize; convert more MOOSE cases with step0 first.
- `--resolution 128` trains much faster; the metrics are then computed at 128.
- The y-mirror augmentation (`--no-flip` disables it) assumes the boundary conditions are symmetric in y.
- Normalization uses per-channel min/max over the training frames only. Predictions are clipped to that range.
- `--context-noise` adds small noise to the context frames during training, which reduces drift in autoregressive rollout.
