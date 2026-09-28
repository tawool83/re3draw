"""Training internals and an end-to-end smoke run.

Everything here needs torch, and the smoke run needs gsplat's CUDA rasteriser, so the whole module
skips on a machine without them. On the GPU box `pytest` therefore checks the training path itself,
not just the data preparation - run it there before spending an hour on a real capture.
"""

import numpy as np
import pytest

from re3draw_worker.colmap import read_colmap
from re3draw_worker.dataset import default_object_box, load_capture
from re3draw_worker.splat import sh_coeffs, write_ply

torch = pytest.importorskip("torch")

from re3draw_worker.train import TrainConfig, evaluate, ssim, to_cloud, train  # noqa: E402
from re3draw_worker.train import _gaussian_window, _init_params  # noqa: E402


@pytest.fixture(scope="module")
def capture(capture_dir):
    spec, root, _ = capture_dir
    return load_capture(read_colmap(root / "sparse" / "0"), root / "images",
                        default_object_box(spec), max_size=256)


def test_ssim_is_one_for_identical_images_and_lower_for_noise():
    window = _gaussian_window(11, 1.5, "cpu", torch.float32)
    rng = np.random.default_rng(0)
    a = torch.tensor(rng.random((1, 3, 64, 64)), dtype=torch.float32)
    assert float(ssim(a, a, window)) == pytest.approx(1.0, abs=1e-4)
    assert float(ssim(a, a.roll(7, dims=-1), window)) < 0.5
    assert 0.0 <= float(ssim(a, torch.zeros_like(a), window)) < 1.0


@pytest.mark.parametrize("degree", [0, 3])
def test_initialised_parameters_have_the_shapes_the_rasteriser_expects(capture, degree):
    cfg = TrainConfig(init_points=500, sh_degree=degree)
    params = _init_params(capture, cfg, "cpu")

    assert set(params) == {"means", "scales", "quats", "opacities", "sh0", "shN"}
    n, k = cfg.init_points, sh_coeffs(degree)
    assert params["means"].shape == (n, 3) and params["quats"].shape == (n, 4)
    assert params["sh0"].shape == (n, 1, 3) and params["shN"].shape == (n, k - 1, 3)
    assert params["opacities"].shape == (n,)
    assert all(p.requires_grad for p in params.values())

    means = params["means"].detach().numpy()
    assert capture.box.contains(means).all(), "initial gaussians must start inside the object box"
    np.testing.assert_allclose(params["quats"].detach().numpy(), np.tile([1, 0, 0, 0], (n, 1)))
    assert float(torch.sigmoid(params["opacities"]).mean()) == pytest.approx(0.1, abs=1e-3)
    assert torch.allclose(params["shN"], torch.zeros_like(params["shN"]))


def test_to_cloud_drops_transparent_gaussians_and_writes_a_ply(capture, tmp_path):
    params = _init_params(capture, TrainConfig(init_points=200), "cpu")
    with torch.no_grad():  # half the population made invisible
        params["opacities"][:100] = -20.0

    cloud = to_cloud(params, min_opacity=0.005)
    assert len(cloud) == 100
    assert cloud.sh_degree == 3
    np.testing.assert_allclose(cloud.means, params["means"].detach().numpy()[100:], rtol=1e-6)
    assert write_ply(cloud, tmp_path / "s.ply").stat().st_size > 0


def test_evaluate_returns_nan_without_views(capture):
    assert np.isnan(evaluate(_init_params(capture, TrainConfig(init_points=50), "cpu"), capture, [], 0, "cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat's rasteriser needs CUDA")
def test_short_training_run_improves_and_exports(capture, tmp_path):
    pytest.importorskip("gsplat")
    cfg = TrainConfig(iterations=300, init_points=5_000, cap_max=20_000, sh_degree=1,
                      val_every=4, log_every=0, seed=0)
    before = evaluate(_init_params(capture, cfg, "cuda"), capture, capture.views[:4], 1, "cuda")
    cloud, metrics = train(capture, cfg, on_log=lambda _: None)

    assert 0 < len(cloud) <= cfg.cap_max
    assert metrics["gaussians_exported"] == len(cloud)
    assert metrics["psnr_val"] > before, f"training did not improve PSNR ({metrics['psnr_val']} vs {before})"
    assert metrics["psnr_val"] > 12.0
    assert metrics["train_views"] + metrics["val_views"] == len(capture.views)
    assert metrics["device"] == "cuda"

    # The box constraint must hold at the end, not just at initialisation.
    assert capture.box.contains(cloud.means).all()
    assert np.isfinite(cloud.means).all() and np.isfinite(cloud.scales).all()
    assert write_ply(cloud, tmp_path / "splat.ply").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat's rasteriser needs CUDA")
def test_training_refuses_an_empty_training_split(capture):
    pytest.importorskip("gsplat")
    with pytest.raises(ValueError, match="no training views"):
        train(capture, TrainConfig(iterations=1, init_points=10, val_every=1))
