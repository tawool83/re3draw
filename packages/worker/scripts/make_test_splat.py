"""Write a splat whose orientation is unmistakable, for viewer tests and demos.

Training needs a GPU, so without this there is nothing for the viewer to load on a laptop. The
shape is deliberately asymmetric in all three axes: any flip, swap or mirror in the viewer's
coordinate handling shows up as the wrong colour pointing the wrong way.

    python scripts/make_test_splat.py ../viewer/test/fixtures/axes.ply

Built in the re3draw world frame: origin at the mat centre, +X right, +Y toward the top edge
(the front camera sits on -Y), +Z up, metres.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from re3draw_worker.splat import GaussianCloud, rgb_to_sh_dc, write_ply  # noqa: E402

AXIS_LENGTH = 0.10  # metres
BEAD = 0.004  # gaussian radius along an axis

# +X red, +Y green, +Z blue - the usual axis colours, so a mismatch is obvious at a glance.
AXES = {
    "x": (np.array([1.0, 0.0, 0.0]), np.array([0.9, 0.15, 0.15])),
    "y": (np.array([0.0, 1.0, 0.0]), np.array([0.15, 0.8, 0.2])),
    "z": (np.array([0.0, 0.0, 1.0]), np.array([0.2, 0.35, 0.95])),
}


def _blob(center, radius, color, count, rng):
    pts = center + rng.normal(0.0, radius / 2.5, size=(count, 3))
    return pts, np.tile(color, (count, 1)), np.full((count, 3), radius / 3.0)


def build(seed: int = 0) -> GaussianCloud:
    rng = np.random.default_rng(seed)
    pos, col, scale = [], [], []

    for direction, color in AXES.values():
        t = np.linspace(0.0, AXIS_LENGTH, 60)[:, None]
        pos.append(t * direction)
        col.append(np.tile(color, (60, 1)))
        scale.append(np.full((60, 3), BEAD))

    # One white cap on +Z and one yellow cap on +Y: distinguishes "up" from "away from the front",
    # which a single axis triad cannot do once it is mirrored.
    for center, color, count in (
        ([0.0, 0.0, AXIS_LENGTH + 0.015], [0.95, 0.95, 0.95], 220),
        ([0.0, AXIS_LENGTH + 0.015, 0.0], [0.95, 0.85, 0.1], 220),
    ):
        p, c, s = _blob(np.array(center), 0.012, np.array(color), count, rng)
        pos.append(p), col.append(c), scale.append(s)

    # A dark disc on the mat plane: marks z = 0 so "standing on the mat" is visible.
    a = rng.uniform(0, 2 * np.pi, 400)
    r = 0.045 * np.sqrt(rng.uniform(0, 1, 400))
    pos.append(np.stack([r * np.cos(a), r * np.sin(a), np.zeros(400)], axis=1))
    col.append(np.tile([0.25, 0.25, 0.28], (400, 1)))
    scale.append(np.full((400, 3), 0.004))

    means = np.vstack(pos)
    colors = np.vstack(col)
    scales = np.vstack(scale)
    n = len(means)
    return GaussianCloud(
        means=means,
        scales=np.log(scales),
        quats=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        opacities=np.full(n, 6.0),  # sigmoid(6) ~ 0.998: opaque, so nothing depends on blending
        sh0=rgb_to_sh_dc(colors).reshape(n, 1, 3),
        shN=np.zeros((n, 0, 3)),  # degree 0 keeps the fixture small
    )


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "axes.ply")
    cloud = build()
    path = write_ply(cloud, out)
    print(f"wrote {path}  ({len(cloud)} gaussians, {path.stat().st_size / 1024:.0f} KB, SH degree {cloud.sh_degree})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
