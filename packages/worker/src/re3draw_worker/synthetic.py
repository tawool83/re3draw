"""Synthetic ring captures of the printed mat with known ground truth (for tests and demos).

Each view ray-casts the printed page on the table plane (Z=0) through a distorted pinhole
camera. The object is either a flat grey cylinder that simply occludes the middle of the mat
(cheap, used by most tests), or a ray-traced :class:`Mug` - hollow, with a handle loop the mat
shows through, a textured and shaded surface - for end-to-end tests where the object has to look
like an object.
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
    image: np.ndarray  # grayscale uint8 (cylinder) or BGR uint8 (mug)
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


def orbit_position(azimuth: float, elevation: float, distance: float) -> np.ndarray:
    """Camera centre on the capture dome. Azimuth 0 = FRONT (-Y), +pi/2 = +X; radians."""
    return np.array([distance * np.cos(elevation) * np.sin(azimuth),
                     -distance * np.cos(elevation) * np.cos(azimuth),
                     distance * np.sin(elevation)])


def default_camera(width: int = 1600, height: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    """Phone-like main camera (~63 deg horizontal FOV) with mild barrel distortion."""
    f = 1.3 * width / 1.6
    K = np.array([[f, 0, width / 2 + 7.3], [0, f, height / 2 - 4.1], [0, 0, 1]])
    return K, np.array([-0.08, 0.03, 0.0, 0.0])


@dataclass(frozen=True)
class Mug:
    """A mug at the mat centre, ray-traced analytically. Metres; the handle is on the +X side.

    Outside: cream with a band of coloured checks (texture for reconstruction to lock onto) and a
    dark round logo on the front (-Y), so front and back can be told apart. Inside: plain cream.
    """

    radius: float = 0.04
    height: float = 0.10
    wall: float = 0.004
    base: float = 0.008
    handle_radius: float = 0.03  # of the arc the tube follows
    tube_radius: float = 0.006
    handle_height: float = 0.055  # centre of the arc
    handle_spheres: int = 48  # the tube is a chain of overlapping spheres

    @property
    def handle_centres(self) -> np.ndarray:
        a = np.radians(np.linspace(-75, 75, self.handle_spheres))
        cx = self.radius - 0.004
        return np.stack([cx + self.handle_radius * np.cos(a), np.zeros_like(a),
                         self.handle_height + self.handle_radius * np.sin(a)], axis=1)

    def trace(self, C: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest hit along rays ``C + t d``: (t, shaded RGB in [0, 1]); t = inf where missed."""
        n = len(d)
        t_hit = np.full(n, np.inf)
        rgb = np.zeros((n, 3))
        # Cheap reject: only rays passing near the mug get the full test.
        centre = np.array([self.radius * 0.3, 0.0, self.height / 2])
        reach = np.hypot(self.radius + 2 * self.handle_radius, self.height / 2) + 0.01
        oc = C - centre
        a = np.einsum("ij,ij->i", d, d)
        b = 2 * d @ oc
        idx = np.nonzero(b * b - 4 * a * (oc @ oc - reach**2) > 0)[0]
        if not len(idx):
            return t_hit, rgb
        dd, aa = d[idx], a[idx]
        best = np.full(len(idx), np.inf)
        normal = np.zeros((len(idx), 3))
        colour = np.zeros((len(idx), 3))

        def take(t, ok, nrm, col):
            ok = ok & (t > 1e-6) & (t < best)
            best[ok] = t[ok]
            normal[ok] = nrm[ok]
            colour[ok] = col[ok] if np.ndim(col) == 2 else col

        def point(t):
            return C + np.nan_to_num(t, nan=-1.0, posinf=-1.0, neginf=-1.0)[:, None] * dd

        # Vertical walls: a quadratic in the xy-plane. Outside wall: entry root; inside wall: exit root.
        a2 = dd[:, 0] ** 2 + dd[:, 1] ** 2
        b2 = 2 * (C[0] * dd[:, 0] + C[1] * dd[:, 1])
        for r, root, lo, outward in ((self.radius, -1, 0.0, True), (self.radius - self.wall, 1, self.base, False)):
            disc = b2 * b2 - 4 * a2 * (C[0] ** 2 + C[1] ** 2 - r * r)
            with np.errstate(invalid="ignore", divide="ignore"):
                t = (-b2 + root * np.sqrt(disc)) / (2 * a2)
            p = point(t)
            ok = (disc > 0) & (p[:, 2] >= lo) & (p[:, 2] <= self.height)
            nrm = np.stack([p[:, 0], p[:, 1], np.zeros(len(p))], axis=1) / r * (1 if outward else -1)
            take(t, ok, nrm, self._outside(p) if outward else MUG_CREAM)

        # Rim (annulus at the top) and inside floor: horizontal planes.
        up = np.tile([0.0, 0.0, 1.0], (len(idx), 1))
        for z, r_in, r_out, col in ((self.height, self.radius - self.wall, self.radius, MUG_CREAM),
                                    (self.base, 0.0, self.radius - self.wall, MUG_CREAM * 0.9)):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (z - C[2]) / dd[:, 2]
            p = point(t)
            rr = np.hypot(p[:, 0], p[:, 1])
            take(t, (rr >= r_in) & (rr <= r_out), up, col)

        # Handle: a chain of spheres along an arc.
        for P in self.handle_centres:
            op = C - P
            bb = 2 * dd @ op
            disc = bb * bb - 4 * aa * (op @ op - self.tube_radius**2)
            with np.errstate(invalid="ignore"):
                t = (-bb - np.sqrt(disc)) / (2 * aa)
            p = point(t)
            take(t, disc > 0, (p - P) / self.tube_radius, MUG_HANDLE)

        hit = np.isfinite(best)
        shade = 0.35 + 0.65 * np.clip(normal @ MUG_LIGHT, 0.0, None)
        t_hit[idx[hit]] = best[hit]
        rgb[idx[hit]] = np.clip(colour[hit] * shade[hit, None], 0, 1)
        return t_hit, rgb

    def _outside(self, p: np.ndarray) -> np.ndarray:
        u = (np.arctan2(p[:, 1], p[:, 0]) / (2 * np.pi)) % 1.0
        v = p[:, 2] / self.height
        cell = (np.floor(u * 24).astype(int) * 7 + np.floor(v * 10).astype(int) * 3) % len(MUG_PALETTE)
        col = np.where(((v > 0.2) & (v < 0.85))[:, None], MUG_PALETTE[cell], MUG_CREAM)
        front = (p[:, 1] < 0) & (np.hypot(p[:, 0], p[:, 2] - self.height * 0.5) < 0.016)
        col[front] = [0.08, 0.08, 0.1]  # the front logo
        return col


MUG_PALETTE = np.array([[0.85, 0.2, 0.2], [0.95, 0.75, 0.15], [0.2, 0.6, 0.3],
                        [0.2, 0.4, 0.85], [0.6, 0.25, 0.65], [0.95, 0.5, 0.2]])
MUG_CREAM = np.array([0.93, 0.9, 0.82])
MUG_HANDLE = np.array([0.15, 0.22, 0.55])
MUG_LIGHT = np.array([-0.4, -0.6, 0.7]) / np.linalg.norm([-0.4, -0.6, 0.7])  # towards the light


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
        return self._camera_noise(img, rng)

    def render_mug(self, R: np.ndarray, t: np.ndarray, rng: np.random.Generator | None, mug: Mug
                   ) -> tuple[np.ndarray, np.ndarray]:
        """(BGR photo, object mask) of the mat with ``mug`` on it; ``rng=None`` means no camera noise."""
        w, h = self.size
        C = -R.T @ t
        d = self.rays @ R
        lam = -C[2] / d[:, 2]
        valid = lam > 0
        pw, ph = self.spec.page_mm
        mx = np.where(valid, ((C[0] + lam * d[:, 0]) * 1000 + pw / 2) / self.mm_per_px - 0.5, -1)
        my = np.where(valid, (ph / 2 - (C[1] + lam * d[:, 1]) * 1000) / self.mm_per_px - 0.5, -1)
        mat = cv2.remap(self.page, mx.reshape(h, w).astype(np.float32), my.reshape(h, w).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=95)
        img = np.repeat(mat.reshape(-1, 1).astype(np.float32), 3, axis=1)
        t_obj, rgb = mug.trace(C, d)
        on_obj = np.isfinite(t_obj) & (~valid | (t_obj < lam))
        img[on_obj] = rgb[on_obj, ::-1] * 255.0  # RGB -> BGR
        img = img.reshape(h, w, 3)
        mask = on_obj.reshape(h, w)
        if rng is None:
            return np.clip(img, 0, 255).astype(np.uint8), mask
        return self._camera_noise(img, rng), mask

    @staticmethod
    def _camera_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
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
    mug: Mug | None = None,
    azimuth_deg: tuple[float, float] | None = None,
) -> tuple[list[SyntheticView], np.ndarray, np.ndarray]:
    """Simulate the ring capture guide around an object at the mat centre.

    ``occluder`` is the (radius, height) in metres of the cylinder standing in for the object;
    ``mug`` replaces it with a ray-traced mug (colour photos). ``azimuth_deg=(lo, hi)`` spreads each
    ring over that arc only (0 = FRONT, +90 = the handle side) instead of all the way round - the
    "only shot the front" case. Returns (views, K, dist); the first view of each ring is the FRONT
    view (camera on -Y) unless ``azimuth_deg`` says otherwise.
    """
    rng = np.random.default_rng(seed)
    K, dist = default_camera(*size)
    renderer = Renderer(spec, K, dist, size)
    views = []
    for ring_idx, (elev_deg, count) in enumerate(rings):
        phi = np.radians(elev_deg)
        for k in range(count):
            if azimuth_deg is None:
                base = 2 * np.pi * k / count
            else:
                lo, hi = np.radians(azimuth_deg)
                base = lo + (hi - lo) * k / max(count - 1, 1)
            theta = base + rng.normal(0, np.radians(2))
            r = distance * rng.uniform(0.95, 1.05)
            C = orbit_position(theta, phi, r)
            target = np.array([rng.normal(0, 0.01), rng.normal(0, 0.01), 0.04])
            R, t = look_at(C, target)
            name = f"r{ring_idx}_{k:02d}.jpg"
            if mug is None:
                views.append(SyntheticView(name, renderer.render(R, t, rng, occluder), R, t, ring_idx,
                                           renderer.object_mask(R, t, occluder)))
            else:
                image, mask = renderer.render_mug(R, t, rng, mug)
                views.append(SyntheticView(name, image, R, t, ring_idx, mask))
    return views, K, dist
