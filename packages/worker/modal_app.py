"""Run re3draw training on Modal's serverless GPUs (billed per second, nothing left running).

    python -m modal run modal_app.py::selftest                        # GPU self-test (pytest)
    python -m modal run modal_app.py --capture ../../out/synth --iters 7000
    python -m modal run modal_app.py --capture ../../out/synth --iters 30000 --gpu L40S

``--capture`` is a local folder as `re3draw-worker pose` leaves it (images/ + sparse/0/). It is
uploaded once to the ``re3draw-captures`` volume, so re-running with other settings does not upload
the photos again. The result lands next to it locally in ``splat-<iters>/`` (splat.ply,
splat.spz, train.json) plus ``modal.json`` with the GPU and billed wall-clock time.

The image uses CUDA 12.4 / PyTorch 2.4 on Python 3.10, the combination gsplat ships prebuilt
wheels for. The worker package itself is mounted from this checkout at container start, so code edits
do not rebuild the image.
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
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "cmake", "build-essential")
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("ninja", "numpy>=1.24", "opencv-contrib-python-headless>=4.8", "pillow>=10.1", "pytest>=7")
    .pip_install("gsplat==1.5.3", extra_index_url="https://docs.gsplat.studio/whl/pt24cu124")
    .pip_install("git+https://github.com/nianticlabs/spz")
    .add_local_python_source("re3draw_worker")
    .add_local_dir(HERE / "tests", "/opt/re3draw/tests")
)

app = modal.App("re3draw-worker", image=image)
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


@app.function(gpu=DEFAULT_GPU, volumes={VOLUME_ROOT: volume}, timeout=60 * 60)
def diagnose(iters: int = 2000, max_size: int = 512) -> None:
    """Training sanity check on a synthetic capture: PSNR on the *same* views before and after,
    the loss log, and render-vs-photo PNGs. Writes to the volume under /_diag/<iters>-<max_size>/,
    so it can run detached:  python -m modal run --detach modal_app.py::diagnose --iters 2000
    """
    import tempfile

    import cv2
    import numpy as np
    import torch

    from re3draw_worker import train as T
    from re3draw_worker.boards import get_board
    from re3draw_worker.colmap import read_colmap, write_colmap
    from re3draw_worker.dataset import default_object_box, load_capture
    from re3draw_worker.pose import estimate_poses
    from re3draw_worker.synthetic import ring_capture

    out = Path(VOLUME_ROOT, "_diag", f"{iters}-{max_size}")
    out.mkdir(parents=True, exist_ok=True)
    spec = get_board("a3")
    views, _, _ = ring_capture(spec, rings=((15.0, 8), (40.0, 8), (70.0, 4)), occluder=(0.06, 0.12))
    res = estimate_poses(spec, [(v.name, v.image) for v in views])
    root = Path(tempfile.mkdtemp())
    (root / "images").mkdir()
    for v in views:
        cv2.imwrite(str(root / "images" / v.name), v.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    write_colmap(res, root / "sparse" / "0")
    cap = load_capture(read_colmap(root / "sparse" / "0"), root / "images", default_object_box(spec),
                       max_size=max_size)

    cfg = T.TrainConfig(iterations=iters, init_points=20_000, cap_max=100_000, sh_degree=1, val_every=4,
                        log_every=max(iters // 20, 1), seed=0)
    train_views, val_views = T.split_views(cap.views, cfg.val_every)
    init = T._init_params(cap, cfg, "cuda")
    report = {
        "gpu": _gpu_name(), "scene_scale": cap.scene_scale, "size": [cap.width, cap.height],
        "mask_fraction": float(np.mean([v.mask.mean() for v in cap.views])),
        "psnr_before": {"train": T.evaluate(init, cap, train_views, 1, "cuda"),
                        "val": T.evaluate(init, cap, val_views, 1, "cuda")},
    }

    trained, log = {}, []
    real_to_cloud = T.to_cloud
    T.to_cloud = lambda params, m: (trained.setdefault("p", params), real_to_cloud(params, m))[1]
    t0 = time.time()
    _, metrics = T.train(cap, cfg, on_log=lambda line: (print(line, flush=True), log.append(line.strip())))
    report["train_seconds"] = round(time.time() - t0, 1)
    report["psnr_after"] = {"train": T.evaluate(trained["p"], cap, train_views, 1, "cuda"),
                            "val": T.evaluate(trained["p"], cap, val_views, 1, "cuda")}
    report["metrics"], report["log"] = metrics, log

    # photo | render | masked photo, for the first training view, before and after training.
    v = train_views[0]
    viewmat, Ks, gt, mask = T._batch(v, cap.K, "cuda")
    for tag, p in (("init", init), ("trained", trained["p"])):
        with torch.no_grad():
            r, _, _ = T._render(p, cap, viewmat, Ks, 1)
        row = torch.cat([gt[0], r[0, ..., :3].clamp(0, 1), gt[0] * mask[0]], dim=1).cpu().numpy()
        cv2.imwrite(str(out / f"{tag}.png"), (row[..., ::-1] * 255).astype(np.uint8))
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    volume.commit()
    print(json.dumps({k: report[k] for k in ("psnr_before", "psnr_after", "train_seconds")}, indent=2))


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
         board: str = "a3", upload: bool = True):
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

    args = ["--iters", str(iters), "--max-size", str(max_size), "--board", board]
    t0 = time.time()
    info = train_remote.with_options(gpu=gpu).remote(name, out_name, args)
    info["call_wall_seconds"] = round(time.time() - t0, 1)  # incl. container start = what is billed
    print(json.dumps(info, indent=2))

    dest = local / out_name
    dest.mkdir(parents=True, exist_ok=True)
    for entry in volume.listdir(f"/{name}/{out_name}"):
        target = dest / Path(entry.path).name
        with open(target, "wb") as f:
            volume.read_file_into_fileobj(entry.path, f)
        print(f"  {target}")
    (dest / "modal.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
