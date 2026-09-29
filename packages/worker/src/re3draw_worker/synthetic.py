"""Synthetic ring captures of the printed mat with known ground truth (for tests and demos).

Each view ray-casts the printed page on the table plane (Z=0) through a distorted pinhole
camera, with a cylinder standing in for the object so it occludes the middle of the mat.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .boards import BoardSpec
from .mat import render_page

# (elevation deg, photo count) for the lower / middle / upper ring of the capture guide.
DEFAULT_RINGS = ((15.0, 18), (40.0, 18), (70.0, 10))


@dataclass
class SyntheticView:
    name: str
    image: np.ndarray  # grayscale uint8
    R: np.ndarray
    t: np.ndarray
    ring: int
    object_mask: np.ndarray | None = None  # (H, W) bool, pixels showing the object: segmentation truth

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


def look_at(center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) for an OpenCV camera at ``center`` looking at ``target`` with +Z up."""
    f = target - center
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    R = np.stack([right, down, f])
    return R, -R @ center


def default_camera(width: int = 1600, height: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    """Phone-like main camera (~63 deg horizontal FOV) with mild barrel distortion."""
    f = 1.3 * width / 1.6
    K = np.array([[f, 0, width / 2 + 7.3], [0, f, height / 2 - 4.1], [0, 0, 1]])
    return K, np.array([-0.08, 0.03, 0.0, 0.0])


class Renderer:
    def __init__(self, spec: BoardSpec, K: np.ndarray, dist: np.ndarray, size: tuple[int, int], dpi: int = 150):
        self.spec, self.K, self.dist, self.size = spec, K, dist, size
        self.mm_per_px = 25.4 / dpi
        page = render_page(spec, dpi)
        self.page = cv2.GaussianBlur(page, (0, 0), 0.8)  # pre-filter against aliasing at far views
        # Normalised ray directions for every (distorted) pixel - shared by all views.
        w, h = size
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pix = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
        norm = cv2.undistortPoints(pix, K, dist).reshape(-1, 2)
        self.rays = np.hstack([norm, np.ones((len(norm), 1))]).astype(np.float64)

    def object_mask(self, R: np.ndarray, t: np.ndarray, occluder=(0.06, 0.12)) -> np.ndarray:
        """Pixels covered by the cylinder standing in for the object (it is convex: hull is exact)."""
        w, h = self.size
        mask = np.zeros((h, w), dtype=np.uint8)
        if occluder:
            radius, height = occluder
            a = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            ring = np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)
            pts = np.vstack([np.hstack([ring, np.zeros((72, 1))]), np.hstack([ring, np.full((72, 1), height)])])
            proj, _ = cv2.projectPoints(pts, cv2.Rodrigues(R)[0], t, self.K, self.dist)
            hull = cv2.convexHull(proj.reshape(-1, 2).astype(np.float32)).astype(np.int32)
            cv2.fillConvexPoly(mask, hull, 1)
        return mask.astype(bool)

    def render(self, R: np.ndarray, t: np.ndarray, rng: np.random.Generator, occluder=(0.06, 0.12)) -> np.ndarray:
        w, h = self.size
        C = -R.T @ t
        d = self.rays @ R  # = (R^T @ rays^T)^T : world ray directions
        lam = -C[2] / d[:, 2]
        valid = lam > 0
        X = C[0] + lam * d[:, 0]
        Y = C[1] + lam * d[:, 1]
        pw, ph = self.spec.page_mm
        mx = np.where(valid, (X * 1000 + pw / 2) / self.mm_per_px - 0.5, -1)
        my = np.where(valid, (ph / 2 - Y * 1000) / self.mm_per_px - 0.5, -1)
        img = cv2.remap(
            self.page, mx.reshape(h, w).astype(np.float32), my.reshape(h, w).astype(np.float32),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=95,  # table
        )

        img[self.object_mask(R, t, occluder)] = 150

        gain = rng.uniform(0.8, 1.1)
        img = cv2.GaussianBlur(img.astype(np.float32) * gain, (0, 0), 0.7)
        img += rng.normal(0, 2.0, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)


def ring_capture(
    spec: BoardSpec,
    rings=DEFAULT_RINGS,
    distance: float = 0.55,
    size: tuple[int, int] = (1600, 1200),
    occluder: tuple[float, float] | None = (0.06, 0.12),
    seed: int = 0,
) -> tuple[list[SyntheticView], np.ndarray, np.ndarray]:
    """Simulate the ring capture guide around an object at the mat centre.

    ``occluder`` is the (radius, height) in metres of the cylinder standing in for the object.
    Returns (views, K, dist). The first view of each ring is the FRONT view (camera on -Y).
    """
    rng = np.random.default_rng(seed)
    K, dist = default_camera(*size)
    renderer = Renderer(spec, K, dist, size)
    views = []
    for ring_idx, (elev_deg, count) in enumerate(rings):
        phi = np.radians(elev_deg)
        for k in range(count):
            theta = 2 * np.pi * k / count + rng.normal(0, np.radians(2))
            r = distance * rng.uniform(0.95, 1.05)
            C = np.array([r * np.cos(phi) * np.sin(theta), -r * np.cos(phi) * np.cos(theta), r * np.sin(phi)])
            target = np.array([rng.normal(0, 0.01), rng.normal(0, 0.01), 0.04])
            R, t = look_at(C, target)
            views.append(SyntheticView(f"r{ring_idx}_{k:02d}.jpg", renderer.render(R, t, rng, occluder), R, t, ring_idx,
                                       renderer.object_mask(R, t, occluder)))
    return views, K, dist
