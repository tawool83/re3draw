"""Which directions around the object were photographed - the capture dome, as cells.

Why: training cannot invent a side nobody photographed, and its held-out score cannot tell either,
because the held-out photos come from the same sides. On the mug benchmark, shooting only the front
half left the back at 13 dB PSNR while training reported 24.6 dB. This check looks at where the
cameras are, before any GPU time is spent.

The dome: azimuth 0 = FRONT (camera on -Y), +90 = the +X side; elevation measured from the mat.
Low and middle rings are split into 8 sectors of 45 deg, the top ring into 4 (near the top,
azimuth matters less). A capture guide app can reuse exactly these cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (name, lowest elevation deg, highest, sectors)
RINGS = (("low", 0.0, 27.5, 8), ("mid", 27.5, 55.0, 8), ("top", 55.0, 90.01, 4))
SECTOR_NAMES = {
    8: ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"],
    4: ["front", "right", "back", "left"],
}
FAIL_GAP_DEG = 180.0  # a whole side never photographed: the model would be half made up
WARN_GAP_DEG = 60.0  # the capture guide steps ~20 deg; three missing steps in a row is visible


def camera_angles(center: np.ndarray) -> tuple[float, float]:
    """(azimuth, elevation) in degrees of a camera centre in the mat frame."""
    x, y, z = center
    return float(np.degrees(np.arctan2(x, -y)) % 360.0), float(np.degrees(np.arctan2(z, np.hypot(x, y))))


def _sector_name(azimuth: float, sectors: int) -> str:
    width = 360.0 / sectors
    return SECTOR_NAMES[sectors][int(((azimuth + width / 2) % 360.0) // width)]


@dataclass
class Coverage:
    counts: dict[str, list[int]]  # ring -> photos per sector, sectors in SECTOR_NAMES order
    empty: list[str]  # "ring/sector" cells without a photo
    max_gap_deg: float  # widest azimuth range with no photo at all, over every ring
    gap_centre: str  # the direction in the middle of that range
    status: str  # "ok" | "warn" | "fail"

    def message(self) -> str:
        if self.status == "fail":
            return (f"no photo within {self.max_gap_deg:.0f} deg around the {self.gap_centre}: that side "
                    "would be made up, not reconstructed. Shoot it, or pass --allow-partial.")
        if self.status == "warn":
            return f"thin coverage - empty: {', '.join(self.empty) or 'none'}; widest gap {self.max_gap_deg:.0f} deg"
        return "every direction photographed"

    def to_json(self) -> dict:
        return {"status": self.status, "max_gap_deg": round(self.max_gap_deg, 1), "gap_centre": self.gap_centre,
                "empty": self.empty, "counts": self.counts, "message": self.message()}


def assess(centers: list[np.ndarray] | np.ndarray) -> Coverage:
    angles = [camera_angles(np.asarray(c, dtype=float)) for c in centers]
    counts, empty = {}, []
    for name, lo, hi, sectors in RINGS:
        row = [0] * sectors
        for az, el in angles:
            if lo <= el < hi:
                row[SECTOR_NAMES[sectors].index(_sector_name(az, sectors))] += 1
        counts[name] = row
        empty += [f"{name}/{SECTOR_NAMES[sectors][i]}" for i, n in enumerate(row) if n == 0]

    az = np.sort([a for a, _ in angles])
    if len(az) == 0:
        return Coverage(counts, empty, 360.0, "front", "fail")
    gaps = np.diff(np.concatenate([az, [az[0] + 360.0]]))
    i = int(np.argmax(gaps))
    max_gap = float(gaps[i])
    centre = _sector_name((az[i] + max_gap / 2) % 360.0, 8)
    status = "fail" if max_gap > FAIL_GAP_DEG else "warn" if (empty or max_gap > WARN_GAP_DEG) else "ok"
    return Coverage(counts, empty, max_gap, centre, status)
