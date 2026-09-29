"""Run re3draw segmentation + training on Modal's serverless GPUs (billed per second).

    python -m modal run modal_app.py::selftest                        # GPU self-test (pytest)
    python -m modal run modal_app.py --capture ../../out/synth --iters 7000
    python -m modal run modal_app.py --capture ../../out/synth --iters 30000 --gpu L40S
    python -m modal run --detach modal_app.py::diagnose --iters 2000   # box-only vs object masks

``--capture`` is a local folder as `re3draw-worker pose` leaves it (images/ + sparse/0/). It is
uploaded once to the ``re3draw-captures`` volume; object masks are then computed there once
(``masks/``) and reused, so re-running with other settings repeats neither. The result lands next
to the capture locally in ``splat-<iters>/`` (splat.ply, splat.spz, train.json, modal.json).

Two images, because SAM 2 needs torch >= 2.5 and gsplat's prebuilt wheels stop at torch 2.4:

* ``train_image``: CUDA 12.4 / torch 2.4 / Python 3.10 (gsplat's wheels are cp310-only) + gsplat
* ``segment_image``: torch 2.6 + transformers, with the SAM 2.1 weights baked in

The worker package is mounted from this checkout at container start, so code edits rebuild neither.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
VOLUME_ROOT = "/data"
DEFAULT_GPU = "L4"

# gsplat's prebuilt Linux wheels are cp310-only, so the image is Python 3.10 with the torch 2.4 /
# CUDA 12.4 wheel (which bundles its own CUDA runtime) rather than the py3.11 pytorch/pytorch image.
train_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "cmake", "build-essential")
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("ninja", "numpy>=1.24", "opencv-contrib-python-headless>=4.8", "pillow>=10.1", "pytest>=7")
    .pip_install("gsplat==1.5.3", extra_index_url="https://docs.gsplat.studio/whl/pt24cu124")
    .pip_install("git+https://github.com/nianticlabs/spz")
    .add_local_python_source("re3draw_worker")
    .add_local_dir(HERE / "tests", "/opt/re3draw/tests")
)

SAM_MODEL = "facebook/sam2.1-hiera-base-plus"


def _download_sam() -> None:
    from transformers import Sam2Model, Sam2Processor

    Sam2Processor.from_pretrained(SAM_MODEL)
    Sam2Model.from_pretrained(SAM_MODEL)


segment_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy>=1.24", "opencv-contrib-python-headless>=4.8", "pillow>=10.1", "transformers>=4.56")
    .env({"HF_HOME": "/models"})
    .run_function(_download_sam)
    .add_local_python_source("re3draw_worker")
)

app = modal.App("re3draw-worker", image=train_image)
volume = modal.Volume.from_name("re3draw-captures", create_if_missing=True)


def _gpu_name() -> str:
    import torch

    return torch.cuda.get_device_name(0)


@app.function(gpu=DEFAULT_GPU, timeout=30 * 60)
def selftest() -> None:
    """The same check setup-gpu.sh ends with: the whole suite, GPU training tests included."""
    import subprocess

    print(f"GPU: {_gpu_name()}", flush=True)
    code = subprocess.run(["python", "-m", "pytest", "/opt/re3draw/tests", "-q", "-rs"]).returncode
    if code:
        raise SystemExit(f"self-test failed (pytest exit {code})")


@app.function(image=segment_image, gpu=DEFAULT_GPU, volumes={VOLUME_ROOT: volume}, timeout=30 * 60)
def segment_remote(capture: str, board: str = "a3", overwrite: bool = False) -> dict:
    """Object masks for a capture on the volume -> <capture>/masks/. Existing masks are kept."""
    from re3draw_worker.boards import get_board
    from re3draw_worker.colmap import read_colmap
    from re3draw_worker.dataset import object_box
    from re3draw_worker.segment import Sam2Segmenter, segment_capture

    volume.reload()
    root = Path(VOLUME_ROOT, capture)
    t0 = time.time()
    report = segment_capture(read_colmap(root / "sparse" / "0"), root / "images", object_box(get_board(board)),
                             root / "masks", Sam2Segmenter(SAM_MODEL), overwrite=overwrite)
    volume.commit()
    return {"kept": len(report["kept"]), "rejected": report["rejected"], "seconds": round(time.time() - t0, 1)}


@app.function(gpu=DEFAULT_GPU, volumes={VOLUME_ROOT: volume}, timeout=60 * 60)
def diagnose(iters: int = 2000, max_size: int = 512) -> None:
    """Does segmentation remove the background curtain? On a synthetic capture with known truth:

    1. SAM 2 masks vs the true object silhouette (IoU per photo);
    2. the same short training with the object box only and with object masks, scored the same way
       (object over black, so background left in the box counts against it);
    3. photo | box-only render | masked render PNGs.

    Writes /_diag/<iters>-<max_size>/ on the volume, so it can run detached.
    """
    import cv2
    import numpy as np
    import torch

    from re3draw_worker import train as T
    from re3draw_worker.boards import get_board
    from re3draw_worker.colmap import read_colmap, write_colmap
    from re3draw_worker.dataset import default_object_box, load_capture
    from re3draw_worker.pose import estimate_poses
    from re3draw_worker.segment import iou, mask_path
    from re3draw_worker.synthetic import ring_capture

    out = Path(VOLUME_ROOT, "_diag", f"{iters}-{max_size}")
    out.mkdir(parents=True, exist_ok=True)
    spec = get_board("a3")
    views, _, _ = ring_capture(spec, rings=((15.0, 8), (40.0, 8), (70.0, 4)), occluder=(0.06, 0.12))
    res = estimate_poses(spec, [(v.name, v.image) for v in views])
    root = Path(VOLUME_ROOT, "_diag", "synth")
    (root / "images").mkdir(parents=True, exist_ok=True)
    for v in views:
        cv2.imwrite(str(root / "images" / v.name), v.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    write_colmap(res, root / "sparse" / "0")
    volume.commit()

    seg = segment_remote.remote("_diag/synth", overwrite=True)
    volume.reload()
    model = read_colmap(root / "sparse" / "0")
    truth = {v.name: v.object_mask for v in views}
    ious = {}
    for e in model.images:  # a rejected photo has no mask: score it as an empty one
        m = cv2.imread(str(mask_path(root / "masks", e.name)), 0)
        ious[e.name] = round(iou(m > 127 if m is not None else np.zeros_like(truth[e.name]), truth[e.name]), 4)

    box = default_object_box(spec)
    masked = load_capture(model, root / "images", box, max_size=max_size, masks_dir=root / "masks")
    box_only = load_capture(model, root / "images", box, max_size=max_size)
    cfg = T.TrainConfig(iterations=iters, init_points=20_000, cap_max=100_000, sh_degree=1, val_every=4,
                        log_every=max(iters // 10, 1), seed=0)
    train_views, val_views = T.split_views(masked.views, cfg.val_every)

    def fit(cap):
        trained = {}
        real_to_cloud = T.to_cloud
        T.to_cloud = lambda params, m: (trained.setdefault("p", params), real_to_cloud(params, m))[1]
        try:
            t0 = time.time()
            T.train(cap, cfg, on_log=lambda line: print(line, flush=True))
        finally:
            T.to_cloud = real_to_cloud
        p = trained["p"]  # scored on the masked views: object over black, identical for both runs
        return p, {"train": round(T.evaluate(p, masked, train_views, 1, "cuda"), 3),
                   "val": round(T.evaluate(p, masked, val_views, 1, "cuda"), 3),
                   "seconds": round(time.time() - t0, 1)}

    p_box, score_box = fit(box_only)
    p_mask, score_mask = fit(masked)

    for i in (0, len(train_views) // 2):
        v = train_views[i]
        viewmat, Ks, gt, _ = T._batch(v, masked.K, "cuda")
        tiles = [gt[0]]
        for p in (p_box, p_mask):
            with torch.no_grad():
                r, _, _ = T._render(p, masked, viewmat, Ks, 1)
            tiles.append(r[0, ..., :3].clamp(0, 1))
        row = torch.cat(tiles, dim=1).cpu().numpy()
        cv2.imwrite(str(out / f"compare_{i:02d}.png"), (row[..., ::-1] * 255).astype(np.uint8))

    report = {"gpu": _gpu_name(), "segment_seconds": seg["seconds"], "segment_rejected": seg["rejected"],
              "iou": {"mean": round(float(np.mean(list(ious.values()))), 4),
                      "min": round(float(np.min(list(ious.values()))), 4), "per_photo": ious},
              "psnr_object_over_black": {"box_only": score_box, "object_masks": score_mask}}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    volume.commit()
    summary = {k: v for k, v in report.items() if k != "iou"}
    summary.update(iou_mean=report["iou"]["mean"], iou_min=report["iou"]["min"])
    print(json.dumps(summary, indent=2))


@app.function(gpu=DEFAULT_GPU, volumes={VOLUME_ROOT: volume}, timeout=2 * 60 * 60, memory=16 * 1024)
def train_remote(capture: str, out_name: str, args: list[str]) -> dict:
    from re3draw_worker.cli import main

    volume.reload()
    capture_dir = f"{VOLUME_ROOT}/{capture}"
    out_dir = f"{capture_dir}/{out_name}"
    t0 = time.time()
    code = main(["train", capture_dir, "-o", out_dir, *args])
    info = {"gpu": _gpu_name(), "exit_code": code, "train_wall_seconds": round(time.time() - t0, 1)}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "modal.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    volume.commit()
    return info


@app.local_entrypoint()
def main(capture: str, iters: int = 30000, gpu: str = DEFAULT_GPU, max_size: int = 1600,
         board: str = "a3", upload: bool = True, segment: bool = True):
    local = Path(capture).resolve()
    if not (local / "images").is_dir() or not (local / "sparse" / "0" / "images.txt").is_file():
        raise SystemExit(f"{local} needs images/ and sparse/0/ - run `re3draw-worker pose` first")
    name = local.name
    out_name = f"splat-{iters}" if gpu == DEFAULT_GPU else f"splat-{iters}-{gpu.lower()}"

    if upload:
        print(f"uploading {local} -> volume re3draw-captures:/{name}")
        with volume.batch_upload(force=True) as batch:
            batch.put_directory(str(local / "images"), f"/{name}/images")
            batch.put_directory(str(local / "sparse"), f"/{name}/sparse")

    t0 = time.time()
    if segment:
        seg = segment_remote.remote(name, board)
        print(f"object masks: {seg['kept']} kept, rejected {seg['rejected'] or 'none'} ({seg['seconds']} s)")
    seg_seconds = round(time.time() - t0, 1)

    args = ["--iters", str(iters), "--max-size", str(max_size), "--board", board]
    if not segment:
        args.append("--no-masks")
    t0 = time.time()
    info = train_remote.with_options(gpu=gpu).remote(name, out_name, args)
    info["call_wall_seconds"] = round(time.time() - t0, 1)  # incl. container start = what is billed
    info["segment_call_seconds"] = seg_seconds
    print(json.dumps(info, indent=2))

    dest = local / out_name
    dest.mkdir(parents=True, exist_ok=True)
    for entry in volume.listdir(f"/{name}/{out_name}"):
        target = dest / Path(entry.path).name
        with open(target, "wb") as f:
            volume.read_file_into_fileobj(entry.path, f)
        print(f"  {target}")
    (dest / "modal.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
