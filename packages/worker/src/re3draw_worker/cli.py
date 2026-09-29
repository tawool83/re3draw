"""re3draw-worker command line.

  re3draw-worker mat   --board a3 -o mat_a3.pdf
  re3draw-worker pose  PHOTOS_DIR --board a3 -o sparse/0
  re3draw-worker synth OUT_DIR --board a3        (synthetic ring capture + ground truth)
  re3draw-worker segment CAPTURE_DIR             (object masks with SAM 2 -> CAPTURE_DIR/masks)
  re3draw-worker train CAPTURE_DIR -o OUT_DIR    (gsplat training -> .ply / .spz; needs a GPU)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .boards import BOARDS, get_board
from .colmap import read_colmap, write_colmap
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


def _cmd_train(args) -> int:
    # Imported here so that `mat`, `pose` and `synth` keep working without torch / gsplat installed.
    from .dataset import load_capture, object_box
    from .splat import SpzUnavailable, write_ply, write_spz
    from .train import BackendMissing, TrainConfig, require_backend, train

    try:
        require_backend()
    except BackendMissing as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    capture_dir = Path(args.capture)
    images_dir = Path(args.images) if args.images else capture_dir / "images"
    sparse_dir = Path(args.sparse) if args.sparse else capture_dir / "sparse" / "0"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model = read_colmap(sparse_dir)
    box = object_box(get_board(args.board), width_m=args.object_width, height_m=args.object_height)
    masks_dir = None if args.no_masks else Path(args.masks) if args.masks else capture_dir / "masks"
    if masks_dir is not None and not masks_dir.is_dir():
        if args.masks:
            print(f"error: {masks_dir} does not exist", file=sys.stderr)
            return 2
        masks_dir = None  # no `segment` run: fall back to the object box alone
    capture = load_capture(model, images_dir, box, max_size=args.max_size, masks_dir=masks_dir)
    print(f"{len(capture.views)} views at {capture.width}x{capture.height}, "
          f"object box {np.round(box.size * 100, 1).tolist()} cm, "
          f"object masks: {masks_dir or 'none (background may leak in)'}")

    cfg = TrainConfig(iterations=args.iters, cap_max=args.cap, sh_degree=args.sh_degree,
                      val_every=args.val_every, seed=args.seed, device=args.device)
    cloud, metrics = train(capture, cfg)

    ply = write_ply(cloud, out / "splat.ply")
    metrics["files"] = {"ply": ply.name}
    if not args.no_spz:
        try:
            metrics["files"]["spz"] = write_spz(ply, out / "splat.spz").name
        except SpzUnavailable as e:
            metrics["spz_skipped"] = str(e)
            print(f"note: {e}")
    metrics["capture"] = {"images": str(images_dir), "sparse": str(sparse_dir),
                          "masks": str(masks_dir) if masks_dir else None}
    (out / "train.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


def _cmd_segment(args) -> int:
    from .dataset import object_box
    from .segment import DEFAULT_MODEL, Sam2Segmenter, segment_capture

    capture_dir = Path(args.capture)
    model = read_colmap(capture_dir / "sparse" / "0")
    box = object_box(get_board(args.board), width_m=args.object_width, height_m=args.object_height)
    report = segment_capture(model, capture_dir / "images", box, capture_dir / "masks",
                             Sam2Segmenter(args.model or DEFAULT_MODEL), overwrite=args.overwrite)
    print(json.dumps({"masks": str(capture_dir / "masks"), "kept": len(report["kept"]),
                      "rejected": report["rejected"]}))
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

    p = sub.add_parser("segment", help="object masks with SAM 2, prompted from the mat -> masks/")
    p.add_argument("capture", help="directory holding images/ and sparse/0/")
    p.add_argument("--board", choices=boards, default="a3")
    p.add_argument("--object-width", type=float, help="object width in metres (default: what the mat supports)")
    p.add_argument("--object-height", type=float, help="object height in metres")
    p.add_argument("--model", help="Hugging Face SAM 2 checkpoint")
    p.add_argument("--overwrite", action="store_true", help="re-segment photos that already have a mask")
    p.set_defaults(func=_cmd_segment)

    p = sub.add_parser("train", help="gsplat training on a posed capture -> .ply / .spz splat")
    p.add_argument("capture", help="directory holding images/ and sparse/0/")
    p.add_argument("--images", help="override the photo directory")
    p.add_argument("--sparse", help="override the COLMAP model directory")
    p.add_argument("--board", choices=boards, default="a3")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--cap", type=int, default=1000000, help="maximum number of gaussians")
    p.add_argument("--max-size", type=int, default=1600, help="longest image side used for training")
    p.add_argument("--sh-degree", type=int, default=3, choices=[0, 1, 2, 3])
    p.add_argument("--val-every", type=int, default=8, help="hold out every Nth photo (0 = train on all)")
    p.add_argument("--object-width", type=float, help="object width in metres (default: what the mat supports)")
    p.add_argument("--object-height", type=float, help="object height in metres")
    p.add_argument("--device", help="cuda, cuda:1, cpu (default: cuda when available)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-spz", action="store_true", help="write only the .ply")
    p.add_argument("--masks", help="object mask directory (default: CAPTURE/masks when it exists)")
    p.add_argument("--no-masks", action="store_true", help="ignore object masks, use the object box only")
    p.set_defaults(func=_cmd_train)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
