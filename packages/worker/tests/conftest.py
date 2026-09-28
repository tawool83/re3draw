"""A synthetic posed capture on disk, shared by the training-input and training tests."""

import cv2
import pytest

from re3draw_worker.boards import get_board
from re3draw_worker.colmap import write_colmap
from re3draw_worker.pose import estimate_poses
from re3draw_worker.synthetic import ring_capture

OCCLUDER = (0.06, 0.12)  # the synthetic object: a 12 cm cylinder of radius 6 cm at the mat centre


@pytest.fixture(scope="session")
def capture_dir(tmp_path_factory):
    """A capture written to disk exactly as `re3draw-worker pose` would leave it."""
    spec = get_board("a3")
    views, _, _ = ring_capture(spec, rings=((15.0, 8), (40.0, 8), (70.0, 4)), occluder=OCCLUDER)
    result = estimate_poses(spec, [(v.name, v.image) for v in views])
    root = tmp_path_factory.mktemp("capture")
    (root / "images").mkdir()
    for v in views:
        cv2.imwrite(str(root / "images" / v.name), v.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    write_colmap(result, root / "sparse" / "0")
    return spec, root, result
