"""The training input path: COLMAP model back in, photos rectified to pinhole, object box masked."""

import cv2
import numpy as np
import pytest

from re3draw_worker.boards import get_board
from re3draw_worker.colmap import read_colmap, write_colmap
from re3draw_worker.dataset import (
    _undistort_map, box_mask, default_object_box, load_capture, object_box,
    random_points_in_box, sample_colors, split_views,
)
from re3draw_worker.pose import detect

from conftest import OCCLUDER


def test_read_colmap_roundtrips_the_written_model(capture_dir):
    _, root, result = capture_dir
    model = read_colmap(root / "sparse" / "0")

    np.testing.assert_allclose(model.camera.K, result.camera.K, atol=1e-4)
    np.testing.assert_allclose(model.camera.dist, result.camera.dist, atol=1e-7)
    assert (model.camera.width, model.camera.height) == (result.camera.width, result.camera.height)
    assert [i.name for i in model.images] == [v.name for v in result.views]
    for entry, view in zip(model.images, result.views):
        np.testing.assert_allclose(entry.R, view.R, atol=1e-7)
        np.testing.assert_allclose(entry.t, view.t, atol=1e-7)
    assert len(model.points) == len({int(i) for v in result.views for i in v.detection.ids})


def test_read_colmap_keeps_images_without_2d_points(capture_dir, tmp_path):
    """COLMAP writes an empty observation line for an image with no points; it must not shift pairing."""
    _, root, result = capture_dir
    src = root / "sparse" / "0"
    dst = tmp_path / "sparse"
    dst.mkdir()
    for name in ("cameras.txt", "points3D.txt"):
        (dst / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")
    lines = (src / "images.txt").read_text(encoding="utf-8").splitlines()
    first_obs = next(i for i, l in enumerate(lines) if not l.startswith("#")) + 1
    lines[first_obs] = ""  # first image now has no 2D points
    (dst / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    model = read_colmap(dst)
    assert [i.name for i in model.images] == [v.name for v in result.views]
    for entry, view in zip(model.images, result.views):
        np.testing.assert_allclose(entry.t, view.t, atol=1e-7)


def test_rectified_photos_are_pinhole(capture_dir):
    """Undistorted images must reproject with the new K and *no* distortion.

    This is the one place where a sign or half-pixel slip in the crop-and-resize bookkeeping would
    otherwise pass unnoticed and quietly smear the trained splat.
    """
    spec, root, _ = capture_dir
    model = read_colmap(root / "sparse" / "0")
    capture = load_capture(model, root / "images", default_object_box(spec), max_size=1600)

    errors = []
    for view in capture.views[:6]:
        gray = cv2.cvtColor(view.image, cv2.COLOR_RGB2GRAY)
        det = detect(spec, view.name, gray)
        assert det is not None, f"mat no longer detected in rectified {view.name}"
        cam = det.world_points @ view.R.T + view.t
        uv = (cam @ capture.K.T)[:, :2] / cam[:, 2:3]
        errors.append(np.sqrt(np.mean(np.sum((uv - det.image_points) ** 2, axis=1))))
    assert np.mean(errors) < 2.0, f"rectified reprojection RMS {np.mean(errors):.2f} px"


def test_max_size_downscales_and_rescales_intrinsics(capture_dir):
    spec, root, _ = capture_dir
    model = read_colmap(root / "sparse" / "0")
    box = default_object_box(spec)
    full = load_capture(model, root / "images", box, max_size=1600)
    small = load_capture(model, root / "images", box, max_size=400)

    assert max(small.width, small.height) <= 400
    ratio = small.width / full.width
    assert small.K[0, 0] / full.K[0, 0] == pytest.approx(ratio, rel=0.02)
    assert (small.K[0, 2] + 0.5) / (full.K[0, 2] + 0.5) == pytest.approx(ratio, rel=0.02)
    assert small.views[0].image.shape[:2] == (small.height, small.width)
    assert small.views[0].mask.shape == (small.height, small.width)


def test_loss_mask_covers_the_object_and_excludes_the_far_mat(capture_dir):
    spec, root, _ = capture_dir
    model = read_colmap(root / "sparse" / "0")
    capture = load_capture(model, root / "images", default_object_box(spec), max_size=800)

    angle = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    radius, height = OCCLUDER
    rim = np.stack([radius * np.cos(angle), radius * np.sin(angle), np.full(48, height)], axis=1)
    object_points = np.vstack([rim, rim * [1, 1, 0], [[0, 0, 0], [0, 0, height]]])

    trimmed = 0
    for view in capture.views:
        cam = object_points @ view.R.T + view.t
        uv = np.round((cam @ capture.K.T)[:, :2] / cam[:, 2:3]).astype(int)
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < capture.width) & (uv[:, 1] >= 0) & (uv[:, 1] < capture.height))
        assert view.mask[uv[inside, 1], uv[inside, 0]].all(), f"{view.name} masks away part of the object"
        assert view.mask.any(), f"{view.name} has an empty loss mask"
        trimmed += view.mask.mean() < 0.9
    assert trimmed > len(capture.views) // 2, "the mask never excludes anything - the box is too large"


def test_box_mask_falls_back_to_everything_when_the_box_is_behind_the_camera():
    box = default_object_box(get_board("a3"))
    R, t = np.eye(3), np.array([0.0, 0.0, -1.0])  # box sits behind the image plane
    assert box_mask(box, R, t, np.eye(3), 32, 24).all()


def test_default_box_matches_the_object_sizes_the_mats_were_measured_with():
    a3, a4 = default_object_box(get_board("a3")), default_object_box(get_board("a4"))
    np.testing.assert_allclose(a3.size, [0.24, 0.24, 0.18])
    np.testing.assert_allclose(a4.size, [0.168, 0.168, 0.126])
    for box in (a3, a4):
        assert box.lower[2] == 0.0  # the object stands on the mat
        np.testing.assert_allclose(box.center[:2], [0.0, 0.0])
        assert len(box.corners) == 8 and box.contains(box.corners).all()
        assert not box.contains(box.upper + 1e-3).any()


def test_object_box_overrides_are_independent():
    spec = get_board("a3")
    box = object_box(spec, width_m=0.05)
    np.testing.assert_allclose(box.size, [0.05, 0.05, default_object_box(spec).size[2]])
    np.testing.assert_allclose(object_box(spec, height_m=0.3).size[2], 0.3)


def test_split_views_holds_out_every_nth(capture_dir):
    spec, root, _ = capture_dir
    capture = load_capture(read_colmap(root / "sparse" / "0"), root / "images",
                           default_object_box(spec), max_size=400)
    train, val = split_views(capture.views, 4)
    assert len(train) + len(val) == len(capture.views)
    assert not {v.name for v in train} & {v.name for v in val}
    assert val[0].name == capture.views[0].name
    assert split_views(capture.views, 0) == (capture.views, [])


def test_sampled_colors_are_plausible(capture_dir):
    spec, root, _ = capture_dir
    capture = load_capture(read_colmap(root / "sparse" / "0"), root / "images",
                           default_object_box(spec), max_size=400)
    box = capture.box
    points = random_points_in_box(box, 2000, seed=3)
    assert box.contains(points).all()

    colors = sample_colors(points, capture)
    assert colors.shape == (2000, 3) and ((colors >= 0) & (colors <= 1)).all()
    # The synthetic scene is grey card and grey object: sampled colours must be near-neutral,
    # which is a weak but real check that pixels are being read at the projected location.
    assert np.abs(colors - colors.mean(axis=1, keepdims=True)).max() < 0.2


def test_load_capture_rejects_photos_that_do_not_match_the_model(capture_dir, tmp_path):
    spec, root, _ = capture_dir
    model = read_colmap(root / "sparse" / "0")
    images = tmp_path / "wrong"
    images.mkdir()
    for entry in model.images[:2]:
        src = cv2.imread(str(root / "images" / entry.name))
        cv2.imwrite(str(images / entry.name), cv2.resize(src, (320, 240)))
    with pytest.raises(ValueError, match="model says"):
        load_capture(model, images, default_object_box(spec))


def test_rectified_photos_have_no_invalid_border(capture_dir):
    """Undistortion must not leave unmapped pixels in the frame.

    A black border is indistinguishable from scene content to the loss, so it would be baked into
    the splat as a dark shell. `alpha=0` buys freedom from that by giving up some field of view.
    """
    spec, root, _ = capture_dir
    model = read_colmap(root / "sparse" / "0")
    map1, map2, _, (x, y, w, h), _ = _undistort_map(model.camera, 1600)

    opaque = np.full((model.camera.height, model.camera.width), 255, np.uint8)
    valid = cv2.remap(opaque, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    assert not (valid[y:y + h, x:x + w] == 0).any(), "rectified frame contains unmapped pixels"

    # The distortion is real, so the rectification must actually be doing something.
    size = (model.camera.width, model.camera.height)
    keep_all, _ = cv2.getOptimalNewCameraMatrix(model.camera.K, model.camera.dist, size, 1.0, size)
    m1, m2 = cv2.initUndistortRectifyMap(model.camera.K, model.camera.dist, None, keep_all, size, cv2.CV_16SC2)
    loose = cv2.remap(opaque, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    assert (loose == 0).sum() > 1000, "no distortion to correct - this test proves nothing"
