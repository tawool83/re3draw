import numpy as np
import pytest

from re3draw_worker.boards import BOARDS, get_board
from re3draw_worker.colmap import rotmat_to_qvec, write_colmap
from re3draw_worker.mat import save_pdf
from re3draw_worker.pose import estimate_poses
from re3draw_worker.synthetic import ring_capture


def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


@pytest.mark.parametrize("name", sorted(BOARDS))
def test_mat_pdf_prints_at_physical_size(tmp_path, name):
    spec = get_board(name)
    path = save_pdf(spec, tmp_path / f"mat_{name}.pdf", dpi=150)
    pdf = path.read_bytes()
    # Pillow writes the page size as /MediaBox [0 0 W H] in points (1/72 inch).
    box = pdf[pdf.index(b"/MediaBox") :].split(b"]")[0].split(b"[")[1].split()
    w_pt, h_pt = float(box[2]), float(box[3])
    assert w_pt == pytest.approx(spec.page_mm[0] / 25.4 * 72, abs=1)
    assert h_pt == pytest.approx(spec.page_mm[1] / 25.4 * 72, abs=1)


def test_board_fits_page_with_margins():
    for spec in BOARDS.values():
        ox, oy = spec.board_origin_mm
        assert ox >= 15 and oy >= 15, spec.name


def test_qvec_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(200):
        A = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        R = A * np.sign(np.linalg.det(A))
        np.testing.assert_allclose(qvec_to_rotmat(rotmat_to_qvec(R)), R, atol=1e-9)


def test_colmap_export(tmp_path):
    spec = get_board("a3")
    views, _, _ = ring_capture(spec, rings=((40.0, 6), (70.0, 4)))
    res = estimate_poses(spec, [(v.name, v.image) for v in views])
    out = write_colmap(res, tmp_path / "sparse")

    cam = (out / "cameras.txt").read_text().splitlines()[-1].split()
    assert cam[1] == "OPENCV" and len(cam) == 12
    assert float(cam[6]) == pytest.approx(res.camera.K[0, 2] + 0.5)

    lines = [l for l in (out / "images.txt").read_text().splitlines() if not l.startswith("#")]
    assert len(lines) == 2 * len(res.views)
    for header, pts, view in zip(lines[::2], lines[1::2], res.views):
        f = header.split()
        np.testing.assert_allclose(qvec_to_rotmat([float(x) for x in f[1:5]]), view.R, atol=1e-6)
        np.testing.assert_allclose([float(x) for x in f[5:8]], view.t, atol=1e-6)
        assert f[9] == view.name
        assert len(pts.split()) == 3 * len(view.detection.ids)

    points = [l for l in (out / "points3D.txt").read_text().splitlines() if not l.startswith("#")]
    assert len(points) == len({int(i) for v in res.views for i in v.detection.ids})
    assert (out / "poses.json").exists()
