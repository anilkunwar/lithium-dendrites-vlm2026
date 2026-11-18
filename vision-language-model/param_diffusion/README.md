# param_diffusion

Conditional diffusion model: (simulation parameters, physical time) → the step0 JPG images of
eta, c, and pot at that time (stacked RGB, 9 channels; choose with `--variables`). Each time
with a JPG for every variable is one training sample. The condition is the
z-scored `params.json` values (log10 for `ko` and `Noise`) plus Fourier features of
log1p(t)/log1p(t_max). An MLP embeds it and adds it to the diffusion timestep embedding of a
diffusers `UNet2DModel`. Training uses v-prediction, a cosine noise schedule, EMA weights, and
condition dropout for classifier-free guidance. Sampling uses DDIM.

Input: `data/<case>/image_files/<var>/<var>_t<time>.jpg` plus `data/<case>/params.json`
(both written by step0). The JPGs are resized to `--size` (default 128) and scaled to [-1, 1].
step0 colors each JPG on its own scale (`tripcolor` without vmin/vmax), so the colors are
relative to the frame. The model learns the images, not absolute field values.

```bash
pip install -r requirements.txt
python train.py --data ../data --output runs/base
python plots.py runs/base
python test.py --run runs/base --case ../data/case_002 --num-samples 4
python test.py --run runs/base --params new_params.json --times 0 25 50 75 100
# outputs: runs/base/test/<name>/images/<var>/<var>_t<time>_s<k>.png
```

## Training metrics (`runs/<name>`)

- `metrics_train.csv`: loss, lr, and gradient norm, averaged over each `--log-every` steps.
- `metrics_eval.csv`: computed every `--eval-every` steps with the EMA weights:
  - `val_loss` and `train_loss_fixed`: diffusion loss with fixed timesteps and noise, so values
    are comparable across steps. A growing gap between them means overfitting.
  - `rmse_<var>` and `psnr_<var>`: RGB error of DDIM samples compared with the simulated JPGs
    on validation samples.
  - `eta_iou` (eta ≥ 0.5) and `front_err` (mean front position / width): computed after the eta
    image is decoded back through the jet colormap.
- `training_curves.png` (redrawn after each evaluation) and `samples/step_*.png`.
- `best_model/` (lowest `val_loss`), `final_model/`, and `checkpoints/last.pt` (`--resume`).

## Validation split

With ≥ 2 cases, whole cases are held out (`--val-cases`, otherwise a random `--val-frac`).
This tests generalization to new parameters. With a single case, the parameters are constant:
the model learns only the time dependence, and validation holds out every `--val-every`-th
frame, which tests interpolation in time only.

A single sample is one draw from a stochastic model: with the noise term, the dendrite
pattern of a simulation is not unique. Use `--num-samples` and compare statistics such as
`front_err`, not just the pixel RMSE.
