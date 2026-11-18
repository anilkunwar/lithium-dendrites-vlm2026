"""Image metrics between generated and simulated JPG images, (B, 3V, H, W) RGB in [0, 1]."""

from __future__ import annotations

import matplotlib
import torch

_JET = None


def jet_to_value(rgb, levels=256):
    """(B, 3, H, W) jet-colored RGB -> (B, H, W) colormap value in [0, 1] (nearest jet color)."""
    global _JET
    if _JET is None:
        try:
            cmap = matplotlib.colormaps["jet"]
        except AttributeError:  # matplotlib < 3.5
            import matplotlib.cm
            cmap = matplotlib.cm.get_cmap("jet")
        _JET = torch.tensor(cmap(torch.linspace(0, 1, levels).numpy())[:, :3], dtype=torch.float32)
    jet = _JET.to(rgb.device)
    B, _, H, W = rgb.shape
    px = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
    idx = torch.cat([torch.cdist(c, jet).argmin(dim=1) for c in px.split(65536)])
    return (idx.float() / (levels - 1)).view(B, H, W)


def metric_names(variables):
    names = [f"rmse_{v}" for v in variables] + [f"psnr_{v}" for v in variables]
    if "eta" in variables:
        names += ["eta_iou", "front_err"]
    return names


def field_metrics(gen, ref, variables, threshold=0.5):
    """Per-sample dict of tensors (B,):

    rmse_<var>, psnr_<var>  RGB error of each variable's image (pixel values in [0, 1])
    eta_iou                 IoU of the deposit masks, eta >= threshold after decoding the jet colors
    front_err               |mean front position| error as a fraction of the width; the front of a row
                            is the fraction of its pixels in the deposit (electrode on the left)
    eta is decoded relative to each image's own color scale (step0 autoscales every frame).
    """
    out = {}
    for k, v in enumerate(variables):
        mse = ((gen[:, 3 * k:3 * k + 3] - ref[:, 3 * k:3 * k + 3]) ** 2).mean(dim=(1, 2, 3))
        out[f"rmse_{v}"] = mse.sqrt()
        out[f"psnr_{v}"] = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
    if "eta" in variables:
        k = list(variables).index("eta")
        g = jet_to_value(gen[:, 3 * k:3 * k + 3]) >= threshold
        r = jet_to_value(ref[:, 3 * k:3 * k + 3]) >= threshold
        inter = (g & r).sum(dim=(-2, -1)).float()
        union = (g | r).sum(dim=(-2, -1)).float()
        out["eta_iou"] = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(union))
        out["front_err"] = (g.float().mean(dim=(-2, -1)) - r.float().mean(dim=(-2, -1))).abs()
    return out
