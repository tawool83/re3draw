"""re3draw-worker command line.

  re3draw-worker mat   --board a3 -o mat_a3.pdf
  re3draw-worker pose  PHOTOS_DIR --board a3 -o sparse/0
  re3draw-worker synth OUT_DIR --board a3        (synthetic ring capture + ground truth)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from .boards import BOARDS, get_board
from .colmap import write_colmap
from .mat import save_pdf
from .pose import PoseError, estimate_poses

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _cmd_mat(args) -> int:
    path = save_pdf(get_board(args.board), args.output, dpi=args.dpi)
    print(f"wrote {path}  (print at 100% / actual size, then check the 100 mm scale bar)")
    return 0


def _cmd_pose(args) -> int:
    spec = get_board(args.board)
    paths = sorted(p for p in Path(args.photos).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    t0 = time.perf_counter()
    images = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)  # applies EXIF orientation
        if img is not None:
            images.append((p.name, img))
    try:
        result = estimate_poses(spec, images)
    except PoseError as e:
        print(json.dumps({"ok": False, "fail_code": e.code, "message": str(e)}))
        return 2
    out = write_colmap(result, args.output)
    print(json.dumps({
        "ok": True, "output": str(out), "views": len(result.views), "rejected": result.rejected,
        "rms_px": round(result.rms_px, 3), "seconds": round(time.perf_counter() - t0, 1),
    }, ensure_ascii=False))
    return 0


def _cmd_synth(args) -> int:
    from .synthetic import ring_capture

    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    views, K, dist = ring_capture(get_board(args.board), seed=args.seed)
    for v in views:
        cv2.imwrite(str(out / "images" / v.name), v.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    gt = {
        "K": K.tolist(), "dist": dist.tolist(),
        "views": [{"name": v.name, "ring": v.ring, "R": v.R.tolist(), "t": v.t.tolist()} for v in views],
    }
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
    print(f"wrote {len(views)} views to {out / 'images'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="re3draw-worker")
    sub = parser.add_subparsers(dest="cmd", required=True)
    boards = sorted(BOARDS)

    p = sub.add_parser("mat", help="write the printable marker mat PDF")
    p.add_argument("--board", choices=boards, default="a3")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=_cmd_mat)

    p = sub.add_parser("pose", help="camera poses from mat photos -> COLMAP text model")
    p.add_argument("photos")
    p.add_argument("--board", choices=boards, default="a3")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=_cmd_pose)

    p = sub.add_parser("synth", help="render a synthetic ring capture with ground truth")
    p.add_argument("output")
    p.add_argument("--board", choices=boards, default="a3")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_synth)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
