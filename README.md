# re3draw

**re·3·draw** — photograph an object from many angles, get a photoreal 3D model (Gaussian splat / mesh),
and show it in Flutter (Android / iOS) and on the web (Three.js).

The trick: camera poses are captured at shooting time (printed ChArUco marker mat, ARCore / ARKit,
LiDAR depth) instead of being guessed from the photos, so the hardest step (SfM) is skipped and the
result has real-world scale.

> Status: **v0.1, milestone M2** — marker mat, camera poses, splat training, and a viewer.
> Verified on synthetic captures only; nothing is published yet.

## Packages

| Path | Package | Status |
| --- | --- | --- |
| [`packages/worker`](packages/worker) | `re3draw-worker` (Python) — mat PDF, poses, COLMAP export, training | M1 ✅ poses · M2 ✅ training |
| [`packages/viewer`](packages/viewer) | `@re3draw/viewer` — Three.js + Spark viewer shared by web and Flutter | M2 ✅ splats |
| [`packages/sdk-js`](packages/sdk-js) | `@re3draw/sdk` — upload / job status / load | M3 |
| [`packages/flutter`](packages/flutter) | `re3draw` — ring capture guide, ARCore / ARKit, LiDAR, viewer widget | M4 |

## Pipeline

```
ring capture (40-50 photos) -> camera poses (mat / ARCore / ARKit) -> COLMAP text model
  -> gsplat training (GPU) -> .spz / .ply splat, .glb mesh -> viewer (Three.js / Flutter)
```

## Try M1

```bash
cd packages/worker
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # macOS/Linux: .venv/bin/pip
re3draw-worker mat --board a3 -o mat_a3.pdf       # print at 100 %, check the 100 mm bar
re3draw-worker synth out/synth                     # or photograph a real object on the mat
re3draw-worker pose out/synth/images --board a3 -o out/synth/sparse/0
# real photos: re3draw-worker prepare PHOTOS_OFF_THE_PHONE -o out/capture1  (shrinks, strips EXIF, poses)
```

## Try M2 (needs an NVIDIA GPU)

```bash
bash packages/worker/scripts/setup-gpu.sh                  # on a rented or local CUDA box
re3draw-worker train out/synth -o out/synth/splat          # -> splat.ply, splat.spz, train.json
```

Training runs on [Modal](https://modal.com) serverless GPUs at roughly 190 won per object; see
[docs/m2-gpu-training.md](docs/m2-gpu-training.md) for setup, options and cost.

## Look at the result

```bash
cd packages/viewer
npm install && npm run dev     # then open your splat.ply / splat.spz with "splat 열기"
```

## License

[Apache-2.0](LICENSE). Only permissively licensed dependencies are allowed (no non-commercial models
such as DUSt3R / MASt3R).
