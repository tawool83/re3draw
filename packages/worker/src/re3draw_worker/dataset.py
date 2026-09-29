"""Turn a posed capture (COLMAP model + photos) into training views.

Three things happen here that are specific to re3draw, and all three come from knowing the mat:

1. **Undistortion.** The rasteriser is a pinhole renderer, so lens distortion has to be removed from
   the photos rather than modelled during training.
2. **A metric object box.** Because the mat fixes real-world scale and the object sits at the
   origin, we know where the object is *before* training. Gaussians are confined to that box.
3. **A loss mask.** Only the part of each photo covered by that box is compared against the render,
   so the room outside it never becomes something training has to explain.

Optionally (``masks_dir``, written by :mod:`.segment`) each view also gets an **object mask**. Inside
the box it separates the object from the background seen *behind* it, which the box alone cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .boards import BoardSpec
from .colmap import ColmapModel

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ObjectBox:
    """Axis-aligned world-space bounds of the object, metres."""

    lower: np.ndarray  # (3,)
    upper: np.ndarray  # (3,)

    @property
    def corners(self) -> np.ndarray:
        """(8, 3) corners, ordered by the bit pattern of (x, y, z) choosing lower/upper."""
        return np.array([[self.lower[0] if i & 1 else self.upper[0],
                          self.lower[1] if i & 2 else self.upper[1],
                          self.lower[2] if i & 4 else self.upper[2]] for i in range(8)])

    @property
    def size(self) -> np.ndarray:
        return self.upper - self.lower

    @property
    def center(self) -> np.ndarray:
        return (self.lower + self.upper) / 2.0

    def contains(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        return np.all((p >= self.lower) & (p <= self.upper), axis=-1)


def default_object_box(spec: BoardSpec) -> ObjectBox:
    """The largest object the mat can pose, as measured in ``tests/test_pose_synthetic.py``.

    Wider than the mat and the corners needed for pose estimation are hidden; much taller and the
    upper ring starts looking down at the object instead of past it.
    """
    short_mm = min(spec.board_mm)
    half = short_mm / 2.0 / 1000.0
    return ObjectBox(np.array([-half, -half, 0.0]), np.array([half, half, short_mm * 0.75 / 1000.0]))


def object_box(spec: BoardSpec, width_m: float | None = None, height_m: float | None = None) -> ObjectBox:
    """Object box, defaulting each dimension to what the mat supports."""
    box = default_object_box(spec)
    half = box.size[0] / 2.0 if width_m is None else width_m / 2.0
    top = box.upper[2] if height_m is None else height_m
    return ObjectBox(np.array([-half, -half, 0.0]), np.array([half, half, top]))


@dataclass(frozen=True)
class View:
    """One undistorted photo with its pose and loss mask."""

    name: str
    image: np.ndarray  # (H, W, 3) uint8, RGB
    mask: np.ndarray  # (H, W) bool - pixels the loss is computed on
    R: np.ndarray  # world -> camera rotation
    t: np.ndarray  # world -> camera translation
    fg: np.ndarray | None = None  # (H, W) float32 in [0, 1] - object, when segmented; None = unknown

    @property
    def viewmat(self) -> np.ndarray:
        """(4, 4) world -> camera transform."""
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = self.R, self.t
        return m

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass(frozen=True)
class Capture:
    """Everything training needs: pinhole intrinsics, views, and the object box."""

    K: np.ndarray  # (3, 3) pinhole intrinsics of the undistorted, resized images
    width: int
    height: int
    views: list[View]
    box: ObjectBox

    @property
    def scene_scale(self) -> float:
        """Radius of the camera rig - the natural length unit for learning rates."""
        centers = np.array([v.center for v in self.views])
        return float(np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1)))


def _undistort_map(camera, max_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int], float]:
    """Build the shared undistort + crop + resize that turns photos into pinhole images.

    ``alpha=0`` keeps only the region that is valid in every direction, so the result has no black
    border for training to interpret as scene content.
    """
    size = (camera.width, camera.height)
    new_K, roi = cv2.getOptimalNewCameraMatrix(camera.K, camera.dist, size, 0.0, size)
    map1, map2 = cv2.initUndistortRectifyMap(camera.K, camera.dist, None, new_K, size, cv2.CV_16SC2)
    x, y, w, h = roi
    if w == 0 or h == 0:  # degenerate ROI (near-zero distortion on some OpenCV builds)
        x, y, w, h = 0, 0, camera.width, camera.height
    scale = min(1.0, max_size / max(w, h))

    K = new_K.copy()
    K[0, 2] -= x
    K[1, 2] -= y
    # Scaling a pixel grid moves sample centres, not just coordinates: (c + 0.5) * s - 0.5.
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] = (K[0, 2] + 0.5) * scale - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * scale - 0.5
    return map1, map2, K, (x, y, w, h), scale


def box_mask(box: ObjectBox, R: np.ndarray, t: np.ndarray, K: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixels covered by the object box, as the convex hull of its projected corners.

    Gaussians live inside the box, so they can only ever paint inside this hull; masking the loss to
    it means the rest of the photo is neither fitted nor fought over.
    """
    cam = box.corners @ R.T + t
    if np.any(cam[:, 2] <= 1e-6):  # a corner behind the camera: projection is meaningless
        return np.ones((height, width), dtype=bool)
    uv = (cam @ K.T)[:, :2] / cam[:, 2:3]
    hull = cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
    return mask.astype(bool)


def load_capture(
    model: ColmapModel,
    images_dir: str | Path,
    box: ObjectBox,
    max_size: int = 1600,
    masks_dir: str | Path | None = None,
) -> Capture:
    """Load and rectify every posed photo. Photos named in the model but missing on disk are skipped.

    With ``masks_dir``, photos with a ``<name>.png`` there get it as their object mask, put through
    the same undistortion, crop and resize as the photo. Photos without one (segmentation rejected
    them) train on the object box alone.
    """
    images_dir = Path(images_dir)
    map1, map2, K, (x, y, w, h), _ = _undistort_map(model.camera, max_size)
    out_w, out_h = 0, 0
    views: list[View] = []
    for entry in model.images:
        path = images_dir / entry.name
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        if (bgr.shape[1], bgr.shape[0]) != (model.camera.width, model.camera.height):
            raise ValueError(f"{entry.name} is {bgr.shape[1]}x{bgr.shape[0]}, model says "
                             f"{model.camera.width}x{model.camera.height}")
        rect = _rectify(bgr, map1, map2, (x, y, w, h), max_size)
        out_h, out_w = rect.shape[:2]
        rgb = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
        mask = box_mask(box, entry.R, entry.t, K, out_w, out_h)
        fg = None
        if masks_dir is not None:
            m = cv2.imread(str(Path(masks_dir) / f"{entry.name}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                fg = _rectify(m, map1, map2, (x, y, w, h), max_size).astype(np.float32) / 255.0 * mask
        views.append(View(entry.name, rgb, mask, entry.R, entry.t, fg))

    if not views:
        raise ValueError(f"none of the {len(model.images)} posed photos were found in {images_dir}")
    return Capture(K=K, width=out_w, height=out_h, views=views, box=box)


def _rectify(img: np.ndarray, map1, map2, roi: tuple[int, int, int, int], max_size: int) -> np.ndarray:
    x, y, w, h = roi
    rect = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)[y:y + h, x:x + w]
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        rect = cv2.resize(rect, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return rect


def split_views(views: list[View], val_every: int) -> tuple[list[View], list[View]]:
    """Hold out every ``val_every``-th photo for evaluation; ``val_every <= 0`` trains on all."""
    if val_every <= 0:
        return list(views), []
    train = [v for i, v in enumerate(views) if i % val_every != 0]
    val = [v for i, v in enumerate(views) if i % val_every == 0]
    return train, val


def sample_colors(points: np.ndarray, capture: Capture) -> np.ndarray:
    """Mean colour of each point over the views that see it, in [0, 1]; grey where nothing sees it.

    Gaussians start at random positions inside the box, so this is not a true surface colour - it is
    a cheap way to start each region of the volume near the right hue instead of at grey.
    """
    points = np.asarray(points, dtype=np.float64)
    total = np.zeros((len(points), 3))
    count = np.zeros(len(points), dtype=np.int64)
    for v in capture.views:
        cam = points @ v.R.T + v.t
        in_front = cam[:, 2] > 1e-6
        uv = np.full((len(points), 2), -1.0)
        uv[in_front] = (cam[in_front] @ capture.K.T)[:, :2] / cam[in_front, 2:3]
        px = np.round(uv).astype(np.int64)
        ok = in_front & (px[:, 0] >= 0) & (px[:, 0] < capture.width) & (px[:, 1] >= 0) & (px[:, 1] < capture.height)
        ok[ok] &= v.mask[px[ok, 1], px[ok, 0]] if v.fg is None else v.fg[px[ok, 1], px[ok, 0]] > 0.5
        total[ok] += v.image[px[ok, 1], px[ok, 0]] / 255.0
        count += ok
    seen = count > 0
    colors = np.full((len(points), 3), 0.5)
    colors[seen] = total[seen] / count[seen, None]
    return colors


def random_points_in_box(box: ObjectBox, count: int, seed: int = 0) -> np.ndarray:
    """Uniform random points inside the box - the training seed, in place of an SfM point cloud."""
    rng = np.random.default_rng(seed)
    return rng.uniform(box.lower, box.upper, size=(count, 3))
