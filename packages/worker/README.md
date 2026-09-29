# re3draw-worker

Python side of re3draw: printable marker mat, camera poses from photos, COLMAP export, and
gaussian splat training.

```bash
pip install -e ".[dev]"
re3draw-worker mat   --board a3 -o mat_a3.pdf
re3draw-worker pose  PHOTOS_DIR --board a3 -o sparse/0
re3draw-worker synth OUT_DIR --board a3           # synthetic ring capture + ground_truth.json
pytest
```

`mat`, `pose` and `synth` run anywhere. `train` needs an NVIDIA GPU and is installed separately:

```bash
bash scripts/setup-gpu.sh                         # on a CUDA machine; see docs/m2-gpu-training.md
re3draw-worker train CAPTURE_DIR -o CAPTURE_DIR/splat
```

## Marker mats

| Board | Page | Grid | Square / marker | Largest object (tested) | Shooting distance |
| --- | --- | --- | --- | --- | --- |
| `a4` | A4 landscape | 9 × 6 | 28 / 21 mm | ~8 cm wide, 10 cm tall | ~40 cm |
| `a3` | A3 landscape | 12 × 8 | 30 / 22.5 mm | ~20 cm wide, 18 cm tall | 55–65 cm |

The object hides the middle of the mat, so the limit is "enough corners still visible". A 12 cm
object on the A4 mat loses half of the views; use A3 for anything larger than ~8 cm.
Print at 100 % / actual size and check the 100 mm scale bar — scale errors become model scale errors.

## World frame

Origin at the centre of the mat, **+X** to the right edge, **+Y** to the top edge, **+Z** up, metres.
The `FRONT` arrow is on the bottom edge: the front photo is taken from **−Y**.

## `pose` output (`sparse/0/`)

- `cameras.txt` — one shared `OPENCV` camera (fx fy cx cy k1 k2 p1 p2), COLMAP pixel convention
- `images.txt` — world→camera pose per photo + detected mat corners as 2D observations
- `points3D.txt` — mat corners as metric sparse points (training seed)
- `poses.json` — summary: RMS error, camera centres, rejected photos with reasons

Failure codes (`fail_code`): `too_few_views` (fewer than 8 photos with the mat),
`calibration_unstable` (fewer than 3 photos from ≥ 30° elevation — the low ring alone cannot
separate lens principal point from camera tilt), `pnp_failed`.
Per-photo rejections: `marker_not_found`, `image_size_mismatch`, `reprojection_error_*`.

## Accuracy (synthetic ring capture, 46 photos: 18 @ 15°, 18 @ 40°, 10 @ 70°)

| Scenario | Photos used | RMS reprojection | Max camera position error | Time |
| --- | --- | --- | --- | --- |
| A3, 12 cm object, 1600×1200 | 45 / 46 | 0.46 px | 2.2 mm | ~2 s |
| A3, 12 cm object, 12 MP | 44 / 46 | 1.49 px (0.03 % of diag.) | 1.5 mm | ~5 s |
| A4, 8 cm object | 46 / 46 | 0.48 px | 2.0 mm | ~2 s |

`tests/test_pose_synthetic.py` enforces F2: RMS < 1 px, ≥ 90 % of photos posed, ≥ 80 % per ring,
position < 6 mm, rotation < 0.5°, under 60 s.

## `train` output

```
re3draw-worker train CAPTURE_DIR -o OUT   # CAPTURE_DIR holds images/ and sparse/0/
```

- `splat.ply` — the 3DGS interchange format every viewer reads
- `splat.spz` — Niantic's format, ~15x smaller, for shipping to phones (needs the optional encoder).
  The encoder writes SPZ **version 4**, which `@re3draw/viewer` cannot display yet — Spark reads
  versions 1-3 only. Load the `.ply` in the viewer until that is resolved.
- `train.json` — PSNR on held-out photos, gaussian count, object box, wall-clock time

Training uses gsplat's **MCMC** densification. The usual adaptive-density strategy grows gaussians
from a structure-from-motion point cloud; we skip SfM entirely, so training starts from random
points and MCMC is what makes that converge.

Two things fall out of knowing the mat, and neither is available to a generic splat pipeline:

- **A metric object box.** The object stands at the origin at a known scale, so its bounds are known
  before training. Gaussians are clamped inside them and cannot escape into the room.
- **A masked loss.** Each photo is compared against the render only inside the projection of that
  box, so the background is never something training has to explain.

Photos are undistorted to a shared pinhole camera first (`cv2` `alpha=0`, cropped to the region
valid in every direction) and downscaled to `--max-size`, since the rasteriser has no lens model.

### Installing the optional pieces

| Piece | Install | Without it |
| --- | --- | --- |
| torch + gsplat | `scripts/setup-gpu.sh` (or `pip install -e ".[train]"`) | `train` is unavailable |
| `.spz` encoder | `pip install "git+https://github.com/nianticlabs/spz"` (MIT) | `train` writes `.ply` only |

`pytest` skips the training tests when torch or CUDA is missing, so the suite stays green on a
laptop and covers the training path on the GPU box.
