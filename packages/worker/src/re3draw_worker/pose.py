"""Camera poses from photos of the marker mat: ChArUco detection -> calibration / PnP -> refinement."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .boards import BoardSpec

MIN_CORNERS = 8
MIN_SELF_CALIB_VIEWS = 8
MIN_STEEP_VIEWS = 3
STEEP_ELEVATION_DEG = 30.0
# Distortion model: k1, k2 radial only (tangential and k3 fixed to 0) - stable on phone photos.
CALIB_FLAGS = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3


@dataclass
class Detection:
    name: str
    image_size: tuple[int, int]  # (width, height)
    ids: np.ndarray  # (N,) ChArUco corner ids
    image_points: np.ndarray  # (N, 2) pixels
    world_points: np.ndarray  # (N, 3) metres


@dataclass
class Camera:
    width: int
    height: int
    K: np.ndarray  # (3, 3)
    dist: np.ndarray  # (4,) k1, k2, p1, p2  (COLMAP OPENCV model)


@dataclass
class ViewPose:
    detection: Detection
    R: np.ndarray  # world -> camera rotation
    t: np.ndarray  # world -> camera translation (metres)
    rms_px: float

    @property
    def name(self) -> str:
        return self.detection.name

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass
class PoseResult:
    camera: Camera
    views: list[ViewPose]
    rejected: dict[str, str] = field(default_factory=dict)  # name -> reason

    @property
    def rms_px(self) -> float:
        sq = [v.rms_px**2 * len(v.detection.ids) for v in self.views]
        n = sum(len(v.detection.ids) for v in self.views)
        return float(np.sqrt(sum(sq) / n)) if n else float("nan")


class PoseError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def detect(spec: BoardSpec, name: str, image: np.ndarray) -> Detection | None:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(spec.board).detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    ids = ids.ravel().astype(np.int32)
    if spec.board.checkCharucoCornersCollinear(ids):
        return None
    return Detection(
        name=name,
        image_size=(gray.shape[1], gray.shape[0]),
        ids=ids,
        image_points=corners.reshape(-1, 2).astype(np.float64),
        world_points=spec.corner_world_points[ids],
    )


def _view_rms(det: Detection, camera: Camera, rvec: np.ndarray, tvec: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(det.world_points, rvec, tvec, camera.K, camera.dist)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - det.image_points) ** 2, axis=1))))


def _calibrate(dets: list[Detection]) -> tuple[Camera, list, list]:
    w, h = dets[0].image_size
    _, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [d.world_points.astype(np.float32) for d in dets],
        [d.image_points.astype(np.float32) for d in dets],
        (w, h), None, None, flags=CALIB_FLAGS,
    )
    return Camera(w, h, K, dist.ravel()[:4].copy()), list(rvecs), list(tvecs)


def _solve_pnp(det: Detection, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    ok, rvec, tvec = cv2.solvePnP(det.world_points, det.image_points, camera.K, camera.dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise PoseError("pnp_failed", det.name)
    return cv2.solvePnPRefineLM(det.world_points, det.image_points, camera.K, camera.dist, rvec, tvec)


def estimate_poses(
    spec: BoardSpec,
    images: list[tuple[str, np.ndarray]],
    camera: Camera | None = None,
    outlier_frac: float = 0.001,
) -> PoseResult:
    """Estimate one shared camera model and a pose per photo.

    With ``camera=None`` intrinsics are self-calibrated from the mat (joint refinement of
    intrinsics + all poses = bundle adjustment with known structure). Views whose RMS
    reprojection error exceeds max(outlier_frac x image diagonal, 3 x median)
    (2 px at 1600x1200, 5 px at 12 MP) are dropped and the fit repeated.
    """
    rejected: dict[str, str] = {}
    dets: list[Detection] = []
    for name, img in images:
        det = detect(spec, name, img)
        if det is None:
            rejected[name] = "marker_not_found"
        elif dets and det.image_size != dets[0].image_size:
            rejected[name] = "image_size_mismatch"
        else:
            dets.append(det)

    min_views = 1 if camera is not None else MIN_SELF_CALIB_VIEWS

    def fit(dets: list[Detection]):
        if len(dets) < min_views:
            raise PoseError("too_few_views", f"{len(dets)} usable photos (need >= {min_views})")
        if camera is None:
            cam, rvecs, tvecs = _calibrate(dets)
        else:
            cam = camera
            rvecs, tvecs = zip(*(_solve_pnp(d, cam) for d in dets))
        errs = [_view_rms(d, cam, r, t) for d, r, t in zip(dets, rvecs, tvecs)]
        return cam, rvecs, tvecs, errs

    cam, rvecs, tvecs, errs = fit(dets)
    for _ in range(2):
        limit = max(outlier_frac * float(np.hypot(*dets[0].image_size)), 3.0 * float(np.median(errs)))
        bad = {i for i, e in enumerate(errs) if e > limit}
        if not bad:
            break
        for i in bad:
            rejected[dets[i].name] = f"reprojection_error_{errs[i]:.1f}px"
        dets = [d for i, d in enumerate(dets) if i not in bad]
        cam, rvecs, tvecs, errs = fit(dets)

    if camera is None:
        # Grazing views alone cannot separate principal point from camera tilt (the fit stays
        # sub-pixel while heights drift by centimetres), so self-calibration needs steep views.
        centers = [-cv2.Rodrigues(r)[0].T @ np.ravel(t) for r, t in zip(rvecs, tvecs)]
        steep = sum(c[2] / np.linalg.norm(c) >= np.sin(np.radians(STEEP_ELEVATION_DEG)) for c in centers)
        if steep < MIN_STEEP_VIEWS:
            raise PoseError(
                "calibration_unstable",
                f"{steep} photos from >= {STEEP_ELEVATION_DEG:.0f} deg elevation (need >= {MIN_STEEP_VIEWS}); "
                "shoot the middle and upper rings or pass known intrinsics",
            )

    views = [
        ViewPose(d, cv2.Rodrigues(r)[0], np.asarray(t, dtype=np.float64).ravel(), e)
        for d, r, t, e in zip(dets, rvecs, tvecs, errs)
    ]
    return PoseResult(cam, views, rejected)
