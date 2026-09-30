"""First step of every capture: turn phone photos as they come off the phone into pipeline input.

Users do not resize photos, and a phone's 12-48 MP JPEGs at 3-4 MB each (140+ MB per capture) are
mostly thrown away downstream anyway: training works at 1600 px on the long side, SAM 2 at 1024.
So the library normalises every photo, the same way on every path (CLI, worker, and later the
web/Flutter SDKs before upload):

* **long side <= 2048 px** (never enlarged) - headroom over training's 1600 px for pose estimation;
  provisional until the resolution benchmark sets it from measurements;
* **JPEG quality 90, no metadata** - drops EXIF entirely, GPS included (privacy requirement);
* **sensor orientation** - the EXIF rotation flag is ignored, so portrait and landscape shots of the
  same phone come out with the same pixel layout and share one camera model. Phones that rotate the
  pixels themselves instead of setting the flag still produce mixed sizes; `pose` rejects those.

Pose estimation must run on the normalised photos, because the intrinsics it estimates are those of
the images training will read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MAX_LONG_SIDE = 2048
JPEG_QUALITY = 90
READABLE = {".jpg", ".jpeg", ".png"}
UNSUPPORTED = {".heic", ".heif"}  # iPhone default; needs a HEIF decoder we do not ship yet


@dataclass
class IngestReport:
    written: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)  # name -> reason
    bytes_in: int = 0
    bytes_out: int = 0
    size_out: tuple[int, int] | None = None  # (width, height) of the first photo written

    def to_json(self) -> dict:
        return {"written": len(self.written), "skipped": self.skipped, "size": self.size_out,
                "mb_in": round(self.bytes_in / 1e6, 1), "mb_out": round(self.bytes_out / 1e6, 1)}


def normalize(image: np.ndarray, max_long_side: int = MAX_LONG_SIDE) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_long_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def read_sensor_oriented(path: Path) -> np.ndarray | None:
    """Pixels as the sensor recorded them (EXIF rotation ignored), BGR."""
    data = np.fromfile(str(path), dtype=np.uint8)  # np.fromfile: non-ASCII paths on Windows
    return cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def ingest(raw_dir: str | Path, images_dir: str | Path, max_long_side: int = MAX_LONG_SIDE) -> IngestReport:
    """Normalise every photo in ``raw_dir`` into ``images_dir`` as ``<stem>.jpg``."""
    raw_dir, images_dir = Path(raw_dir), Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    report = IngestReport()
    for path in sorted(p for p in raw_dir.iterdir() if p.is_file()):
        ext = path.suffix.lower()
        if ext in UNSUPPORTED:
            report.skipped[path.name] = "heic_unsupported"
            continue
        if ext not in READABLE:
            continue
        image = read_sensor_oriented(path)
        if image is None:
            report.skipped[path.name] = "unreadable"
            continue
        out = normalize(image, max_long_side)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            report.skipped[path.name] = "encode_failed"
            continue
        target = images_dir / f"{path.stem}.jpg"
        buf.tofile(str(target))
        report.written.append(target.name)
        report.bytes_in += path.stat().st_size
        report.bytes_out += len(buf)
        report.size_out = report.size_out or (out.shape[1], out.shape[0])
    return report
