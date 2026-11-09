# Frame diffusion: predicting future phase-field frames

A conditional diffusion model that takes the previous K frames (`eta, c, pot`) and generates
the next frame. Repeating this autoregressively produces a sequence. It is built from
standard components:

- [HuggingFace diffusers](https://github.com/huggingface/diffusers): `UNet2DModel`, `DDPMScheduler`
  for training, `DDIMScheduler` for sampling, and `EMAModel`
- [accelerate](https://github.com/huggingface/accelerate): mixed precision, multiple GPUs, checkpoint resume
- Conditioning: the context frames are concatenated with the noisy target along the channel
  axis (as in MCVD, Voleti et al. 2022, and SR3). MOOSE uses adaptive time steps, so the log time
  gaps between frames are embedded and added to the timestep embedding. The sampled simulation
  parameters (`data/<case>/params.json`, z-scored over the training cases, in log10 for parameters sampled
  over decades) are embedded the same way, with an optional null embedding for classifier-free guidance.

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
python frame_diffusion/train.py --data data --output runs --context 4
```

Each run creates its own directory `runs/<YYYYmmdd-HHMMSS>_k4_params` (`_noparams` with `--no-params`),
so earlier runs are never overwritten. After training, the best model is tested automatically
(autoregressive rollout of up to 50 frames per held-out case, written to `<run>/test/`).
To resume, pass only the checkpoint; the run's saved settings are reused:

```bash
python frame_diffusion/train.py --resume runs/20261020-153012_k4_params/checkpoint-5000
```

Training length is chosen from the validation score: every 1000 steps (`--val-every`) the EMA
model predicts the next frame for 256 fixed held-out windows, and the mean squared error
(`val_mse`) is recorded. Whenever it reaches a new low, the weights are saved to `model/`. Training stops
after 10 validations without improvement. `--max-steps` (200k) is only an upper bound.
The learning rate is constant after warmup, as in the diffusers examples.

```bash
python frame_diffusion/test.py --run runs/<run>
```

Train/test split:

- `--val-cases case_005 case_006`: whole held-out cases (preferred once more cases are converted)
- Otherwise each case is split in time. Windows whose target is in the first `--train-frac`
  (0.8) of the frames are used for training, and the test rolls out over the remaining frames.
  This tests extrapolation to later growth stages.

Both step0 scripts copy `generated_inputs/<case>.json` to `data/<case>/params.json`. `--no-params`
trains without them; compare the two runs to see how much the parameters help.

Predict the frames after a given time with given parameters:

```bash
python frame_diffusion/test.py --run runs/<run> --case case_066 --t0 150 --num-frames 10
```

```bash
python frame_diffusion/test.py --run runs/<run> --case case_066 --t0 150 --num-frames 10 --param POT_LEFT=-0.4 --output runs/<run>/test_pot04
```

The context is the 4 simulation frames at or before `t0`. The parameters come from the case's own JSON,
from `--params-file`, and from any `--param NAME=VALUE` overrides; `summary.json` records the values used.
With changed parameters, the metrics still compare against the original simulation, so they measure the
difference from it, not accuracy.

### From parameters and one frame

To generate the frames after a given frame for a given set of parameters, train with one context frame on a
uniform time grid and with parameter dropout (the defaults `--param-dropout 0.1` and log10 scaling of
`fo ko L1o L2o Noise delta` apply whenever parameters are used):

```bash
python frame_diffusion/train.py --data data --output runs --context 1 --dt 5
```

- `--dt 5` interpolates every case linearly onto t0, t0+5, ... (MOOSE saves at adaptive steps, about 6.5
  apart for case_001), so each predicted frame is 5 time units after the previous one. `test.py` uses the
  same grid.
- `--param-names POT_LEFT L1o L2o ko Noise` conditions on a subset of the keys in `params.json`.
- `--param-dropout` replaces the parameters with a learned null embedding in 10% of the training samples.
  Sampling with `--guidance w` (w > 1) then strengthens the effect of the parameters
  (classifier-free guidance); `--guidance 1` is plain conditional sampling.

```bash
python frame_diffusion/test.py --run runs/<run> --init-frames data/case_001/npy_files/0.000000.npy \
    --params-file my_params.json --num-frames 20 --samples 4 --guidance 2 --name trial_01
```

`--init-frames` takes K `(H, W, 3)` npy files (oldest first) instead of a case; there is no ground truth, so
the outputs are `sample_XX/npy_files`, `prediction.png` (ensemble mean and std), `front.json` (tip position and
area fraction per frame and sample), and `summary.json`. `my_params.json` holds the conditioning parameters by
name (a case's `params.json` works); `--param NAME=VALUE` overrides single values.

### Validating the surrogate (with Qwen)

```bash
python frame_diffusion/validate_surrogate.py --run runs/<run> --samples 8
```

The simulations are noisy, so `validate_surrogate.py` judges the model statistically. For every test case it draws
`--samples` rollouts and compares step1's morphology metrics with the MOOSE frames. Code decides pass/fail for:
final-frame class agreement, error relative to the MOOSE seed-to-seed spread (needs cases that differ only in the
seed), interval coverage, eta bounds, Li inventory, and (with `--sweep POT_LEFT`) the direction of a parameter's
effect. It also reports the trusted rollout horizon. Then Qwen (Ollama, `--qwen-model`) compares MOOSE and samples
side by side (its answers are checked against the metrics), takes a blind test (which row is the simulation?), and
writes a verdict whose cited evidence keys and overall rating are checked against the computed criteria.
Results go to `<run>/validation/`: `report.md`, `summary.json`, `summary.png`, `cases/`, `qwen/`.
`--no-qwen` skips Qwen; `--qwen-only` reruns only the Qwen part on saved results. Qwen's verdict is a draft:
check `qwen/reading_checks.json` first, which shows how often its reading of the images matched the metrics.

Useful `test.py` options:

| Option | Purpose |
| --- | --- |
| `--teacher-forcing` | Condition every step on ground-truth frames (one-step error, no accumulation) |
| `--samples 8` | Multiple stochastic rollouts; reports the ensemble mean and standard deviation |
| `--case case_001 --start 10 --num-frames 30` | Choose the starting frame and length |
| `--dt 6.5` | Time step for frames beyond the end of the data |
| `--steps 50` | DDIM sampling steps |

## Outputs

`runs/<name>/`: `model/` (EMA weights at the lowest val_mse, `from_pretrained`), `scheduler/`, `run_config.json`
(arguments, normalization statistics, split), `train_log.jsonl`, and `checkpoint-*` (pass to `--resume`).

Refreshed at every validation during training:

- `train_curves.png`: training loss (with a moving average) and val_mse, with the best step marked.
  `python frame_diffusion/plots.py runs/<name>` redraws it at any time.
- `val_samples/step_XXXXXX.png`: 4 fixed validation windows. Columns: last context frame (η), then the
  ground truth, prediction, and absolute error for each channel (normalized units). Compare steps to see the
  predictions improve.

`runs/<name>/test/<case>/`:

- `sample_XX/npy_files/<time>.npy`: predicted frames in the same `(H, W, 3)` format as step0
- `metrics.json`: per frame and channel MSE/PSNR/SSIM, plus tip position (fraction of domain width)
  and area fraction of the deposit attached to the left boundary (`eta >= 0.5`, 4-connectivity)
- `comparison.png`: ground truth, prediction, and error; `metrics.png`: metric curves over time
- `rollout.gif`: animation of the rollout, ground truth on top and prediction below for each channel

## Notes

- Currently only `case_001` (56 frames, 40 training windows) is converted. That is enough to check
  the pipeline but not to generalize; convert more MOOSE cases with step0 first.
- `--resolution 128` trains much faster; the metrics are then computed at 128.
- The y-mirror augmentation (`--no-flip` disables it) assumes the boundary conditions are symmetric in y.
- Normalization uses per-channel min/max over the training frames only. Predictions are clipped to that range.
- `--context-noise` adds small noise to the context frames during training, which reduces drift in autoregressive rollout.

## Exploring parameters with Qwen

```bash
python frame_diffusion/explore_agent.py --run runs/<run> --base-case case_005 --axes POT_LEFT L2o --goal-file goals/uniform_growth.json
```

`explore_agent.py` lets Qwen plan experiments on the surrogate. Each step Qwen sees a table of all evaluated
points (p = fraction of rollouts meeting the goal, classes, final metrics, onset), the logistic boundary fitted by
code (with the power-law exponent when both axes are log-scaled) and a phase plot, and answers with one action:
`grid`, `points`, `refine` (code places points near the boundary), `test_invariance` (is an expression such as
`L2o*abs(POT_LEFT)/L1o` the controlling quantity? points on lines of constant value, AUC against each single
parameter), `propose_moose`, or `finish`. Points outside the training range are rejected; the rollout length
defaults to the trusted horizon from `validate_surrogate.py`. Results go to `<run>/explore/<stamp>_<axes>/`:
`report.md`, `phase.png`, `session.json`, `moose_candidates.json`, and every request in `iter_XX/`.
`--resume <dir>` continues a session (with a new `--budget` / `--max-iters` if given). Goal files are in `goals/`;
`--target-class uniform --min-front 0.3` is a shortcut. Qwen's conclusions are drafts until the MOOSE candidates are run.
