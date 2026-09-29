"""Segmentation prompt, mask checks and object-mask plumbing, against the synthetic ground truth.

SAM 2 itself needs its own GPU environment (see modal_app.py); what is tested here is everything
around it that decides whether it gets a good prompt and whether its masks reach training intact.
"""

import json

import cv2
import numpy as np
import pytest

from re3draw_worker.boards import get_board
from re3draw_worker.colmap import read_colmap, write_colmap
from re3draw_worker.dataset import default_object_box, load_capture
from re3draw_worker.pose import estimate_poses
from re3draw_worker.segment import REPORT, box_region, check_mask, iou, mask_path, seed_point, segment_capture
from re3draw_worker.synthetic import ring_capture


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    spec = get_board("a3")
    views, _, _ = ring_capture(spec, rings=((15.0, 6), (40.0, 6), (70.0, 4)), occluder=(0.06, 0.12))
    result = estimate_poses(spec, [(v.name, v.image) for v in views])
    root = tmp_path_factory.mktemp("seg")
    (root / "images").mkdir()
    for v in views:
        cv2.imwrite(str(root / "images" / v.name), v.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    write_colmap(result, root / "sparse" / "0")
    truth = {v.name: v.object_mask for v in views}
    return spec, root, read_colmap(root / "sparse" / "0"), truth


def test_seed_point_lands_on_the_object_from_every_ring(synthetic):
    _, _, model, truth = synthetic
    for entry in model.images:
        x, y = np.round(seed_point(model, entry)).astype(int)
        assert truth[entry.name][y, x], entry.name


def test_object_lies_inside_the_projected_box(synthetic):
    spec, _, model, truth = synthetic
    for entry in model.images:
        region = box_region(model, entry, default_object_box(spec))
        assert check_mask(truth[entry.name], region) is None, entry.name


def test_check_mask_rejects_nothing_and_the_table():
    region = np.zeros((100, 100), bool)
    region[40:60, 40:60] = True
    assert check_mask(np.zeros_like(region), region) == "empty"
    table = np.ones_like(region)  # SAM grabbed the whole scene
    assert check_mask(table, region).startswith("outside_object_box")
    assert check_mask(region.copy(), region) is None


class TruthSegmenter:
    """Stands in for SAM 2 and returns the ground truth (or a bad mask for chosen photos).
    Photos are told apart by their seed point, which is unique per camera pose."""

    def __init__(self, model, truth, bad=()):
        self.by_seed = {tuple(np.round(seed_point(model, e), 3)): (e.name, truth[e.name]) for e in model.images}
        self.bad, self.calls = set(bad), []

    def __call__(self, rgb, point):
        name, mask = self.by_seed[tuple(np.round(point, 3))]
        self.calls.append(name)
        return np.zeros_like(mask) if name in self.bad else mask


def test_masks_survive_rectification_into_training_views(synthetic, tmp_path):
    spec, root, model, truth = synthetic
    box = default_object_box(spec)
    bad = model.images[0].name
    report = segment_capture(model, root / "images", box, tmp_path / "masks", TruthSegmenter(model, truth, [bad]))
    assert report["rejected"] == {bad: "empty"}
    assert set(report["kept"]) == {e.name for e in model.images} - {bad}
    assert json.loads((tmp_path / "masks" / REPORT).read_text()) == report
    assert not mask_path(tmp_path / "masks", bad).exists()

    capture = load_capture(model, root / "images", box, max_size=800, masks_dir=tmp_path / "masks")
    for view in capture.views:
        if view.name == bad:
            assert view.fg is None, "a rejected photo trains on the object box alone"
            continue
        assert view.fg is not None and view.fg.shape == view.mask.shape
        assert (view.fg[~view.mask] == 0).all(), "object mask must be limited to the object box"
        # The object is the flat 150-grey cylinder: masked pixels should be that grey.
        grey = view.image[view.fg > 0.99].mean()
        assert abs(grey - 150) < 25, grey


def test_existing_masks_are_not_recomputed(synthetic, tmp_path):
    spec, root, model, truth = synthetic
    box = default_object_box(spec)
    first = segment_capture(model, root / "images", box, tmp_path / "masks", TruthSegmenter(model, truth))
    again = TruthSegmenter(model, truth)
    assert segment_capture(model, root / "images", box, tmp_path / "masks", again) == first
    assert again.calls == []


def test_iou():
    a = np.zeros((4, 4), bool)
    a[:2] = True
    b = np.zeros((4, 4), bool)
    b[1:3] = True
    assert iou(a, b) == pytest.approx(1 / 3)
    assert iou(a, a) == 1.0
