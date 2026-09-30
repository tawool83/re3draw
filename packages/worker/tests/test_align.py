"""Reading splats back, moving them, and fitting a foreign asset onto the mat by its silhouettes."""

import numpy as np
import pytest

from re3draw_worker.splat import GaussianCloud, read_ply, rotmat_to_quat, transformed, write_ply


def lopsided_cloud(n=4000, seed=0, sh_rest=0):
    """An L-shaped blob: no symmetry, so up axis, yaw and scale are all recoverable."""
    rng = np.random.default_rng(seed)
    a = rng.uniform([-0.5, -0.1, -0.1], [0.5, 0.1, 0.1], (n // 2, 3))  # long bar along x
    b = rng.uniform([0.3, -0.1, 0.1], [0.5, 0.1, 0.6], (n - n // 2, 3))  # post up at one end (+z)
    means = np.vstack([a, b])
    return GaussianCloud(
        means=means, scales=np.full((n, 3), np.log(0.02)), quats=np.tile([1.0, 0, 0, 0], (n, 1)),
        opacities=np.full(n, 3.0), sh0=rng.normal(size=(n, 1, 3)), shN=rng.normal(size=(n, sh_rest, 3)),
    )


@pytest.mark.parametrize("sh_rest", [0, 15])
def test_ply_roundtrip(tmp_path, sh_rest):
    cloud = lopsided_cloud(500, sh_rest=sh_rest)
    back = read_ply(write_ply(cloud, tmp_path / "c.ply"))
    for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        np.testing.assert_array_equal(getattr(back, name), getattr(cloud, name), err_msg=name)


def test_transformed_moves_scales_and_rotates():
    cloud = lopsided_cloud(100)
    c, s = np.cos(0.7), np.sin(0.7)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    out = transformed(cloud, R, 0.25, np.array([0.1, -0.2, 0.0]))
    np.testing.assert_allclose(out.means, 0.25 * cloud.means @ R.T + [0.1, -0.2, 0.0], atol=1e-6)
    np.testing.assert_allclose(out.scales, cloud.scales + np.log(0.25), atol=1e-6)
    np.testing.assert_allclose(np.abs(out.quats), np.abs(rotmat_to_quat(R))[None].repeat(100, 0), atol=1e-6)


def _cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda(), reason="gsplat's rasteriser needs CUDA")
def test_fit_recovers_up_axis_yaw_and_size():
    pytest.importorskip("gsplat")
    import torch
    from gsplat import rasterization

    from re3draw_worker.align import UP_AXES, Silhouette, apply, fit_to_silhouettes
    from re3draw_worker.synthetic import look_at, orbit_position

    # The "generated" asset lies on its side: its up direction is +y, as some generators emit.
    asset = transformed(lopsided_cloud(), UP_AXES["+y"].T, 1.0, np.zeros(3))
    # Truth: stand it up, turn it 70 deg, make it 12 cm long, put it on the mat centre.
    c, s = np.cos(np.radians(70)), np.sin(np.radians(70))
    yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    world = transformed(asset, yaw @ UP_AXES["+y"], 0.12, np.zeros(3))
    lift = -np.percentile(world.means[:, 2], 1)
    world = transformed(world, np.eye(3), 1.0, np.array([0, 0, lift]))

    w, h = 200, 150
    K = np.array([[180.0, 0, w / 2], [0, 180.0, h / 2], [0, 0, 1]])
    views = []
    for az, el in [(0, 30), (90, 30), (180, 30), (270, 30), (0, 75)]:
        R, t = look_at(orbit_position(np.radians(az), np.radians(el), 0.4), np.array([0, 0, 0.04]))
        vm = np.eye(4)
        vm[:3, :3], vm[:3, 3] = R, t
        params = [torch.tensor(x, dtype=torch.float32, device="cuda") for x in
                  (world.means, world.quats, np.exp(world.scales), 1 / (1 + np.exp(-world.opacities)))]
        _, a, _ = rasterization(*params, torch.ones((len(world), 3), device="cuda"),
                                torch.tensor(vm[None], dtype=torch.float32, device="cuda"),
                                torch.tensor(K[None], dtype=torch.float32, device="cuda"), w, h, packed=False)
        views.append(Silhouette(vm, K, a[0, ..., 0].cpu().numpy() > 0.5))

    fit = fit_to_silhouettes(asset, views)
    assert fit.iou > 0.9, fit
    placed = apply(asset, fit)
    # Same shape in the same place: compare the clouds' extents and centroids.
    np.testing.assert_allclose(np.percentile(placed.means, [5, 95], axis=0),
                               np.percentile(world.means, [5, 95], axis=0), atol=0.01)
