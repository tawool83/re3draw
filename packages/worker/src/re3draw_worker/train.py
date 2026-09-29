"""Gaussian splat training on a posed capture, using gsplat's MCMC densification.

Why MCMC rather than the original adaptive-density strategy: that one grows gaussians by splitting
the ones that already have large view-space gradients, which assumes a sparse point cloud from
structure-from-motion to start from. We deliberately skip SfM, so training starts from random
points inside the object box. MCMC treats the gaussian set as a fixed-size population that it keeps
relocating, which is what makes a random start converge.

torch and gsplat are imported lazily so that the rest of the CLI works on machines without a GPU
stack installed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .dataset import Capture, View, random_points_in_box, sample_colors, split_views
from .splat import GaussianCloud, rgb_to_sh_dc, sh_coeffs

if TYPE_CHECKING:  # pragma: no cover
    import torch


@dataclass
class TrainConfig:
    iterations: int = 30_000
    cap_max: int = 1_000_000  # ceiling on the gaussian population (MCMC keeps it fixed once reached)
    init_points: int = 100_000
    sh_degree: int = 3
    sh_degree_interval: int = 1_000  # raise the active SH degree this often
    val_every: int = 8  # hold out every Nth photo; 0 trains on all of them
    ssim_weight: float = 0.2
    opacity_reg: float = 0.01  # MCMC needs these to stop it hoarding faint, huge gaussians
    scale_reg: float = 0.01
    min_opacity: float = 0.005
    # Largest gaussian axis as a fraction of the object box's longest side. Positions are clamped
    # to the box, but without this a gaussian at the box edge can stretch far outside it and show
    # up as long streaks across the mat.
    max_scale_frac: float = 0.1
    seed: int = 0
    device: str | None = None
    log_every: int = 500
    # Learning rates from the original 3DGS; the position rate is in scene-radius units and decays.
    lr: dict[str, float] = field(default_factory=lambda: {
        "means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20,
    })
    means_lr_final_ratio: float = 0.01


class BackendMissing(RuntimeError):
    """torch or gsplat is not installed on this machine."""


def require_backend() -> None:
    """Fail before any photo is read, with the instruction the user actually needs."""
    missing = []
    for module in ("torch", "gsplat"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise BackendMissing(
            f"training needs {' and '.join(missing)}, which are not installed. "
            "On a CUDA machine run scripts/setup-gpu.sh; training cannot run on macOS or without "
            "an NVIDIA GPU. See docs/m2-gpu-training.md."
        )
    resolve_device()


def resolve_device(requested: str | None = None) -> str:
    """Pick the training device. gsplat has no CPU rasteriser, so "no GPU" is a hard stop here
    rather than a slow surprise a few thousand iterations in."""
    import torch

    if requested:
        return requested
    if not torch.cuda.is_available():
        raise BackendMissing(
            "torch cannot see a CUDA GPU. gsplat only rasterises on the GPU, so training cannot "
            "run here. See docs/m2-gpu-training.md."
        )
    return "cuda"


def _gaussian_window(size: int, sigma: float, device, dtype):
    import torch

    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(a: "torch.Tensor", b: "torch.Tensor", window: "torch.Tensor") -> "torch.Tensor":
    """Mean SSIM of two (1, 3, H, W) images in [0, 1]."""
    import torch.nn.functional as F

    pad, c1, c2 = window.shape[-1] // 2, 0.01**2, 0.03**2
    mu_a = F.conv2d(a, window, padding=pad, groups=3)
    mu_b = F.conv2d(b, window, padding=pad, groups=3)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = F.conv2d(a * a, window, padding=pad, groups=3) - mu_aa
    var_b = F.conv2d(b * b, window, padding=pad, groups=3) - mu_bb
    cov = F.conv2d(a * b, window, padding=pad, groups=3) - mu_ab
    num = (2 * mu_ab + c1) * (2 * cov + c2)
    den = (mu_aa + mu_bb + c1) * (var_a + var_b + c2)
    return (num / den).mean()


def _init_params(capture: Capture, cfg: TrainConfig, device: str) -> "torch.nn.ParameterDict":
    import torch

    points = random_points_in_box(capture.box, cfg.init_points, seed=cfg.seed)
    colors = sample_colors(points, capture)
    # Uniform points in a box sit, on average, this far apart; starting at that radius means the
    # initial population just covers the volume without swamping it.
    spacing = float(np.cbrt(np.prod(capture.box.size) / max(cfg.init_points, 1)))

    n, k = len(points), sh_coeffs(cfg.sh_degree)
    sh0 = rgb_to_sh_dc(colors).reshape(n, 1, 3)
    t = lambda a, d=torch.float32: torch.tensor(np.ascontiguousarray(a), dtype=d, device=device)  # noqa: E731
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(t(points)),
        "scales": torch.nn.Parameter(t(np.full((n, 3), math.log(spacing)))),
        "quats": torch.nn.Parameter(t(np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)))),
        # logit(0.1): start translucent so overlapping gaussians can blend instead of fighting.
        "opacities": torch.nn.Parameter(t(np.full(n, math.log(0.1 / 0.9)))),
        "sh0": torch.nn.Parameter(t(sh0)),
        "shN": torch.nn.Parameter(t(np.zeros((n, k - 1, 3)))),
    }).to(device)


def _batch(view: View, K: np.ndarray, device: str):
    import torch

    viewmat = torch.tensor(view.viewmat, dtype=torch.float32, device=device)[None]
    Ks = torch.tensor(K, dtype=torch.float32, device=device)[None]
    gt = torch.tensor(view.image, dtype=torch.float32, device=device)[None] / 255.0  # (1, H, W, 3)
    mask = torch.tensor(view.mask, dtype=torch.float32, device=device)[None, ..., None]
    return viewmat, Ks, gt, mask


def _object_target(view: View, gt, render, alpha, bg):
    """With an object mask: composite both sides over ``bg``, so the object is fitted and every
    non-object pixel has to stay transparent to let ``bg`` through. Without one: unchanged."""
    if view.fg is None:
        return render, gt
    import torch

    fg = torch.tensor(view.fg, dtype=torch.float32, device=gt.device)[None, ..., None]
    return render + (1.0 - alpha) * bg, gt * fg + (1.0 - fg) * bg


def _render(params, capture: Capture, viewmat, Ks, sh_degree: int):
    from gsplat import rasterization
    import torch

    colors = torch.cat([params["sh0"], params["shN"]], dim=1)  # (N, K, 3)
    return rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=viewmat,
        Ks=Ks,
        width=capture.width,
        height=capture.height,
        sh_degree=sh_degree,
        packed=False,
    )


def _psnr(render, gt, mask) -> float:
    import torch

    err = ((render - gt) ** 2 * mask).sum() / (mask.sum() * 3).clamp(min=1)
    return float(10.0 * torch.log10(1.0 / err.clamp(min=1e-12)))


def evaluate(params, capture: Capture, views: list[View], sh_degree: int, device: str) -> float:
    """Mean PSNR over ``views``, inside the loss mask (object over black when segmented).
    Returns nan for an empty list."""
    import torch

    if not views:
        return float("nan")
    with torch.no_grad():
        scores = []
        for view in views:
            viewmat, Ks, gt, mask = _batch(view, capture.K, device)
            render, alpha, _ = _render(params, capture, viewmat, Ks, sh_degree)
            black = torch.zeros(3, device=device)
            render, gt = _object_target(view, gt, render[..., :3], alpha, black)
            scores.append(_psnr(render.clamp(0, 1), gt, mask))
    return float(np.mean(scores))


def to_cloud(params, min_opacity: float) -> GaussianCloud:
    """Detach the trained parameters into a numpy cloud, dropping near-invisible gaussians."""
    import torch

    with torch.no_grad():
        keep = torch.sigmoid(params["opacities"]) >= min_opacity
        take = lambda name: params[name][keep].detach().cpu().numpy()  # noqa: E731
        return GaussianCloud(
            means=take("means"), scales=take("scales"), quats=take("quats"),
            opacities=take("opacities"), sh0=take("sh0"), shN=take("shN"),
        )


def train(capture: Capture, cfg: TrainConfig | None = None, on_log=print) -> tuple[GaussianCloud, dict]:
    """Fit a splat to the capture. Returns the cloud and a metrics dict."""
    import torch
    from gsplat.strategy import MCMCStrategy

    cfg = cfg or TrainConfig()
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    started = time.perf_counter()

    train_views, val_views = split_views(capture.views, cfg.val_every)
    if not train_views:
        raise ValueError("no training views left after the validation split")
    scene_scale = capture.scene_scale
    params = _init_params(capture, cfg, device)

    # One optimiser per parameter: MCMC relocates individual gaussians and has to reset exactly
    # those rows of the Adam state, which it can only do if the parameters are not pooled.
    optimizers = {
        name: torch.optim.Adam(
            [{"params": [params[name]], "lr": cfg.lr[name] * (scene_scale if name == "means" else 1.0)}],
            eps=1e-15,
        )
        for name in params
    }
    means_lr0 = cfg.lr["means"] * scene_scale
    decay = cfg.means_lr_final_ratio ** (1.0 / max(cfg.iterations - 1, 1))

    strategy = MCMCStrategy(cap_max=cfg.cap_max, min_opacity=cfg.min_opacity,
                            refine_stop_iter=int(cfg.iterations * 0.83))
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    window = _gaussian_window(11, 1.5, device, torch.float32)
    lower = torch.tensor(capture.box.lower, dtype=torch.float32, device=device)
    upper = torch.tensor(capture.box.upper, dtype=torch.float32, device=device)
    max_log_scale = math.log(cfg.max_scale_frac * float(np.max(capture.box.size)))
    rng = np.random.default_rng(cfg.seed)
    order: list[int] = []

    for step in range(cfg.iterations):
        if not order:
            order = list(rng.permutation(len(train_views)))
        view = train_views[order.pop()]
        sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        means_lr = means_lr0 * decay**step
        optimizers["means"].param_groups[0]["lr"] = means_lr

        viewmat, Ks, gt, mask = _batch(view, capture.K, device)
        render, alpha, info = _render(params, capture, viewmat, Ks, sh_degree)
        # A fresh random background each step: the only way to match it is real transparency.
        render, gt = _object_target(view, gt, render[..., :3], alpha, torch.rand(3, device=device))

        # Both sides are masked, so pixels outside the object box contribute no gradient at all.
        pred, target = render * mask, gt * mask
        pixels = mask.sum() * 3
        l1 = ((pred - target).abs().sum() / pixels.clamp(min=1))
        chw = lambda x: x.permute(0, 3, 1, 2)  # noqa: E731
        loss = (1.0 - cfg.ssim_weight) * l1 + cfg.ssim_weight * (1.0 - ssim(chw(pred), chw(target), window))
        loss = loss + cfg.opacity_reg * torch.sigmoid(params["opacities"]).abs().mean()
        loss = loss + cfg.scale_reg * torch.exp(params["scales"]).abs().mean()

        # No pre-backward hook: that one exists so the adaptive-density strategy can retain
        # view-space gradients, which MCMC does not look at.
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, optimizers, state, step, info, lr=means_lr)
        with torch.no_grad():  # keep the population inside the metric object box, and small enough
            params["means"].data.clamp_(lower, upper)  # not to reach far out of it
            params["scales"].data.clamp_(max=max_log_scale)

        if cfg.log_every and (step + 1) % cfg.log_every == 0:
            on_log(f"  step {step + 1}/{cfg.iterations}  loss {float(loss):.4f}  "
                   f"gaussians {len(params['means']):,}  sh {sh_degree}")

    metrics = {
        "iterations": cfg.iterations,
        "gaussians": int(len(params["means"])),
        "sh_degree": cfg.sh_degree,
        "train_views": len(train_views),
        "val_views": len(val_views),
        "psnr_train": round(evaluate(params, capture, train_views[:8], cfg.sh_degree, device), 3),
        "psnr_val": round(evaluate(params, capture, val_views, cfg.sh_degree, device), 3),
        "scene_scale_m": round(scene_scale, 4),
        "object_box_m": {"lower": capture.box.lower.round(4).tolist(), "upper": capture.box.upper.round(4).tolist()},
        "image_size": [capture.width, capture.height],
        "device": device,
        "seconds": round(time.perf_counter() - started, 1),
    }
    cloud = to_cloud(params, cfg.min_opacity)
    metrics["gaussians_exported"] = len(cloud)
    return cloud, metrics
