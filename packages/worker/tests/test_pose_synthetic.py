"""F2 acceptance on synthetic ring captures: < 1 px reprojection, 50 photos < 1 min on CPU."""

import time

import numpy as np
import pytest

from re3draw_worker.boards import get_board
from re3draw_worker.pose import Camera, PoseError, estimate_poses
from re3draw_worker.synthetic import ring_capture

# (board, object (radius, height) m, shooting distance m): the largest object each mat supports.
SCENARIOS = {
    "a4-8cm": ("a4", (0.04, 0.10), 0.42),
    "a3-12cm": ("a3", (0.06, 0.12), 0.55),
    "a3-20cm": ("a3", (0.10, 0.18), 0.65),
}


def rot_err_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))


@pytest.fixture(scope="module", params=list(SCENARIOS))
def capture(request):
    board, occluder, distance = SCENARIOS[request.param]
    spec = get_board(board)
    views, K, dist = ring_capture(spec, distance=distance, occluder=occluder)
    return spec, views, K, dist


def test_self_calibrated_poses(capture):
    spec, views, K, _ = capture
    t0 = time.perf_counter()
    res = estimate_poses(spec, [(v.name, v.image) for v in views])
    elapsed = time.perf_counter() - t0

    gt = {v.name: v for v in views}
    assert elapsed < 60
    assert res.rms_px < 1.0
    assert len(res.views) >= 0.9 * len(views)
    for ring in range(3):  # every ring must keep usable coverage
        assert sum(gt[v.name].ring == ring for v in res.views) >= 0.8 * sum(v.ring == ring for v in views)
    for v in res.views:
        assert np.linalg.norm(v.center - gt[v.name].center) < 0.006  # metres
        assert rot_err_deg(v.R, gt[v.name].R) < 0.5
    assert abs(res.camera.K[0, 0] / K[0, 0] - 1) < 0.01


def test_known_intrinsics(capture):
    spec, views, K, dist = capture
    w, h = views[0].image.shape[::-1]
    res = estimate_poses(spec, [(v.name, v.image) for v in views], camera=Camera(w, h, K, dist))
    gt = {v.name: v for v in views}
    assert res.rms_px < 1.0
    for v in res.views:
        assert np.linalg.norm(v.center - gt[v.name].center) < 0.006


def test_front_view_is_on_negative_y():
    spec = get_board("a3")
    views, _, _ = ring_capture(spec, rings=((40.0, 8), (70.0, 4)))
    res = estimate_poses(spec, [(v.name, v.image) for v in views])
    front = next(v for v in res.views if v.name == views[0].name)
    assert front.center[1] < -0.2 and front.center[2] > 0


def test_low_ring_only_is_rejected_as_unstable():
    # Grazing views fit at sub-pixel error yet put cameras ~7 cm too low: must not pass silently.
    spec = get_board("a3")
    views, _, _ = ring_capture(spec, rings=((15.0, 18),))
    with pytest.raises(PoseError) as e:
        estimate_poses(spec, [(v.name, v.image) for v in views])
    assert e.value.code == "calibration_unstable"


def test_no_mat_raises():
    spec = get_board("a3")
    blank = [(f"{i}.jpg", np.full((600, 800), 128, np.uint8)) for i in range(5)]
    with pytest.raises(PoseError) as e:
        estimate_poses(spec, blank)
    assert e.value.code == "too_few_views"
