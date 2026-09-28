"""COLMAP text-model export (cameras.txt / images.txt / points3D.txt) + poses.json summary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pose import PoseResult

# OpenCV puts the centre of the top-left pixel at (0, 0); COLMAP puts it at (0.5, 0.5).
_COLMAP_PIXEL_OFFSET = 0.5


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z) with w >= 0."""
    m = np.asarray(R, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = [0.0] * 4
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def write_colmap(result: PoseResult, out_dir: str | Path, pose_source: str = "marker") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cam = result.camera
    fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
    k1, k2, p1, p2 = cam.dist

    (out / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 OPENCV {cam.width} {cam.height} {fx:.6f} {fy:.6f} "
        f"{cx + _COLMAP_PIXEL_OFFSET:.6f} {cy + _COLMAP_PIXEL_OFFSET:.6f} {k1:.8f} {k2:.8f} {p1:.8f} {p2:.8f}\n",
        encoding="utf-8",
    )

    # Mat corners become sparse 3D points (point id = corner id + 1) - a metric seed for training.
    tracks: dict[int, list[tuple[int, int]]] = {}
    world: dict[int, np.ndarray] = {}
    lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "#   POINTS2D[] as (X, Y, POINT3D_ID)"]
    for image_id, view in enumerate(result.views, start=1):
        qw, qx, qy, qz = rotmat_to_qvec(view.R)
        tx, ty, tz = view.t
        lines.append(f"{image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f} 1 {view.name}")
        det = view.detection
        pts = []
        for idx, (cid, (u, v), X) in enumerate(zip(det.ids, det.image_points, det.world_points)):
            pid = int(cid) + 1
            pts.append(f"{u + _COLMAP_PIXEL_OFFSET:.3f} {v + _COLMAP_PIXEL_OFFSET:.3f} {pid}")
            tracks.setdefault(pid, []).append((image_id, idx))
            world[pid] = X
        lines.append(" ".join(pts))
    (out / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    p_lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)"]
    for pid in sorted(tracks):
        X = world[pid]
        track = " ".join(f"{i} {j}" for i, j in tracks[pid])
        p_lines.append(f"{pid} {X[0]:.6f} {X[1]:.6f} {X[2]:.6f} 128 128 128 {result.rms_px:.4f} {track}")
    (out / "points3D.txt").write_text("\n".join(p_lines) + "\n", encoding="utf-8")

    summary = {
        "pose_source": pose_source,
        "units": "metres",
        "world_frame": "origin=mat centre, +X right, +Y toward top edge (front camera on -Y), +Z up",
        "rms_reprojection_px": round(result.rms_px, 4),
        "camera": {
            "model": "OPENCV", "width": cam.width, "height": cam.height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "p1": p1, "p2": p2,
        },
        "views": [
            {"name": v.name, "corners": int(len(v.detection.ids)), "rms_px": round(v.rms_px, 4),
             "center": [round(float(c), 6) for c in v.center]}
            for v in result.views
        ],
        "rejected": result.rejected,
    }
    (out / "poses.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
