"""The `.ply` we hand to viewers (and to the .spz encoder) must match the 3DGS layout exactly."""

import numpy as np
import pytest

from re3draw_worker.splat import SH_C0, GaussianCloud, rgb_to_sh_dc, sh_coeffs, write_ply


def make_cloud(n=5, degree=3, seed=0):
    rng = np.random.default_rng(seed)
    k = sh_coeffs(degree)
    return GaussianCloud(
        means=rng.normal(size=(n, 3)), scales=rng.normal(size=(n, 3)), quats=rng.normal(size=(n, 4)),
        opacities=rng.normal(size=n), sh0=rng.normal(size=(n, 1, 3)), shN=rng.normal(size=(n, k - 1, 3)),
    )


def read_ply(path):
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii").splitlines()
    names = [l.split()[-1] for l in header if l.startswith("property ")]
    assert all(l.split()[1] == "float" for l in header if l.startswith("property "))
    data = np.frombuffer(raw[end:], dtype="<f4").reshape(-1, len(names))
    return header, names, data


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_ply_header_and_property_order(tmp_path, degree):
    cloud = make_cloud(n=7, degree=degree)
    header, names, data = read_ply(write_ply(cloud, tmp_path / "s.ply"))

    assert header[0] == "ply"
    assert header[1] == "format binary_little_endian 1.0"
    assert header[2] == "element vertex 7"
    rest = 3 * (sh_coeffs(degree) - 1)
    assert names == (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
                     + [f"f_rest_{i}" for i in range(rest)]
                     + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    assert data.shape == (7, len(names))


def test_ply_values_roundtrip(tmp_path):
    cloud = make_cloud(n=11)
    _, names, data = read_ply(write_ply(cloud, tmp_path / "s.ply"))
    col = {name: data[:, i] for i, name in enumerate(names)}

    np.testing.assert_allclose(np.stack([col[a] for a in "xyz"], 1), cloud.means, rtol=1e-6)
    np.testing.assert_allclose(np.stack([col[f"scale_{i}"] for i in range(3)], 1), cloud.scales, rtol=1e-6)
    np.testing.assert_allclose(np.stack([col[f"rot_{i}"] for i in range(4)], 1), cloud.quats, rtol=1e-6)
    np.testing.assert_allclose(col["opacity"], cloud.opacities, rtol=1e-6)
    np.testing.assert_allclose(np.stack([col[f"f_dc_{i}"] for i in range(3)], 1), cloud.sh0[:, 0], rtol=1e-6)
    assert np.all(np.stack([col[a] for a in ("nx", "ny", "nz")]) == 0)


def test_f_rest_is_channel_major():
    """3DGS stores every coefficient of R, then of G, then of B - not coefficient by coefficient."""
    n, degree = 3, 2
    rest = sh_coeffs(degree) - 1  # 8 coefficients per channel
    shN = np.arange(n * rest * 3, dtype=np.float32).reshape(n, rest, 3)
    cloud = GaussianCloud(
        means=np.zeros((n, 3)), scales=np.zeros((n, 3)), quats=np.zeros((n, 4)),
        opacities=np.zeros(n), sh0=np.zeros((n, 1, 3)), shN=shN,
    )
    flat = np.transpose(cloud.shN, (0, 2, 1)).reshape(n, -1)
    assert flat[0, 0] == shN[0, 0, 0] and flat[0, 1] == shN[0, 1, 0]  # consecutive = same channel
    assert flat[0, rest] == shN[0, 0, 1]  # the next channel starts after `rest` entries


def test_sh_dc_conversion_matches_viewer_formula():
    rgb = np.array([[0.0, 0.5, 1.0]])
    np.testing.assert_allclose(0.5 + SH_C0 * rgb_to_sh_dc(rgb), rgb, atol=1e-6)


def test_cloud_rejects_inconsistent_shapes():
    with pytest.raises(ValueError, match="quats"):
        GaussianCloud(np.zeros((4, 3)), np.zeros((4, 3)), np.zeros((4, 3)),
                      np.zeros(4), np.zeros((4, 1, 3)), np.zeros((4, 15, 3)))
    with pytest.raises(ValueError, match="degree"):
        GaussianCloud(np.zeros((4, 3)), np.zeros((4, 3)), np.zeros((4, 4)),
                      np.zeros(4), np.zeros((4, 1, 3)), np.zeros((4, 7, 3)))


def test_sh_degree_is_recovered_from_shape():
    for degree in range(4):
        assert make_cloud(degree=degree).sh_degree == degree
