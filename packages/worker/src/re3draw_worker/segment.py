"""Cut the object out of every photo with SAM 2, prompted automatically from the mat.

Why: the loss mask from the object box also covers whatever is *behind* the object (table, wall), and
training then paints that background into the box as a blurry curtain. An object mask per photo lets
training fit the object only and learn that everything else is transparent.

SAM 2 needs a prompt, and the mat provides one for free: the capture guide puts the object on the
centre of the mat, so points 1 and 2 cm above the mat centre are inside the object, and their
projections land on the object in every photo, whatever the angle. No user clicks.

SAM answers a point with three candidates - a part, a larger part, the whole thing - and the one it
is most confident about is often a *part*: on the textured synthetic mug it returned the printed logo
the point happened to land on (IoU with the true silhouette 0.38). We therefore take the **largest
candidate that passes** :func:`check_mask`, i.e. the whole object but not the table. Measured on the
mug benchmark (IoU, mean / worst photo): one point + most confident 0.38 / 0.06; one point + largest
valid 0.92 / 0.51; two points (1, 2 cm) + largest valid 0.94 / 0.75. Earlier, on a plain cylinder,
prompting with the projected object box gave 0.00-0.23 (SAM returns the mat) and mat corners as
negative points made it worse. Points are kept low so objects down to ~3 cm tall still contain them.

Masks are checked before they are kept: an empty mask, or one reaching well outside the object box,
means SAM picked something else (the table, the mat) and that photo trains without a mask. Specks
far smaller than the object are dropped.

Masks are written at the original photo resolution as ``masks/<photo name>.png`` (255 = object), so
training at any resolution reuses them. SAM 2 wants torch >= 2.5 while gsplat's prebuilt wheels stop
at torch 2.4, so segmentation runs in its own environment (see modal_app.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .colmap import ColmapModel, ModelImage
from .dataset import ObjectBox

DEFAULT_MODEL = "facebook/sam2.1-hiera-base-plus"
SEED_HEIGHTS_M = (0.01, 0.02)  # prompt points above the mat centre, i.e. inside the object
MIN_OBJECT_FRACTION = 0.0005  # of the photo; smaller means SAM found nothing
MIN_INSIDE_BOX = 0.9  # share of the mask that must lie inside the projected object box
MIN_COMPONENT = 0.02  # connected pieces smaller than this share of the mask are specks
REPORT = "report.json"


def mask_path(masks_dir: str | Path, image_name: str) -> Path:
    return Path(masks_dir) / f"{image_name}.png"


def _project(model: ColmapModel, image: ModelImage, points: np.ndarray) -> np.ndarray:
    cam = model.camera
    uv, _ = cv2.projectPoints(np.asarray(points, dtype=np.float64), cv2.Rodrigues(image.R)[0], image.t,
                              cam.K, cam.dist)
    return uv.reshape(-1, 2)


def seed_points(model: ColmapModel, image: ModelImage) -> np.ndarray:
    """(N, 2) positive prompts: the mat centre raised into the object, in original photo pixels."""
    return _project(model, image, [[0.0, 0.0, z] for z in SEED_HEIGHTS_M])


def box_region(model: ColmapModel, image: ModelImage, box: ObjectBox) -> np.ndarray:
    """(H, W) bool: the projected object box, in original photo pixels."""
    cam = model.camera
    hull = cv2.convexHull(_project(model, image, box.corners).astype(np.float32))
    region = np.zeros((cam.height, cam.width), dtype=np.uint8)
    cv2.fillConvexPoly(region, np.round(hull).astype(np.int32), 1)
    return region.astype(bool)


def check_mask(mask: np.ndarray, region: np.ndarray) -> str | None:
    """None if the mask looks like the object, else why it was rejected."""
    area = int(mask.sum())
    if area < MIN_OBJECT_FRACTION * mask.size:
        return "empty"
    inside = float(np.logical_and(mask, region).sum()) / area
    if inside < MIN_INSIDE_BOX:
        return f"outside_object_box_{1 - inside:.0%}"
    return None


def choose_mask(candidates: np.ndarray, scores: np.ndarray, region: np.ndarray) -> np.ndarray:
    """The largest candidate that passes :func:`check_mask`, else SAM's most confident one (which the
    caller's check will then reject)."""
    valid = [m for m in candidates if check_mask(m, region) is None]
    if valid:
        return max(valid, key=lambda m: int(m.sum()))
    return candidates[int(np.argmax(scores))]


def drop_specks(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.nonzero(areas >= MIN_COMPONENT * areas.sum())[0] + 1
    return np.isin(labels, keep)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


class Sam2Segmenter:
    """SAM 2.1 through Hugging Face transformers (Apache-2.0 code and weights)."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, rgb: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """SAM's candidates for the object under ``points``: ((K, H, W) bool masks, (K,) scores)."""
        import torch

        inputs = self.processor(images=rgb, input_points=[[[[float(x), float(y)] for x, y in points]]],
                                input_labels=[[[1] * len(points)]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=True)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
        return masks.reshape(-1, *rgb.shape[:2]).numpy().astype(bool), out.iou_scores.cpu().numpy().ravel()


def segment_capture(
    model: ColmapModel,
    images_dir: str | Path,
    box: ObjectBox,
    masks_dir: str | Path,
    segmenter=None,
    overwrite: bool = False,
) -> dict:
    """Write ``masks_dir/<name>.png`` for every posed photo whose mask passes :func:`check_mask`.

    Returns and writes ``masks_dir/report.json``: object share of each kept photo, reason for each
    rejected one. Photos that already have a mask are skipped unless ``overwrite``.
    """
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    report = {"kept": {}, "rejected": {}}
    if (masks_dir / REPORT).exists() and not overwrite:  # keep what earlier runs decided
        report = json.loads((masks_dir / REPORT).read_text(encoding="utf-8"))
    for entry in model.images:
        out = mask_path(masks_dir, entry.name)
        if out.exists() and not overwrite:
            continue
        bgr = cv2.imread(str(images_dir / entry.name), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        segmenter = segmenter or Sam2Segmenter()
        region = box_region(model, entry, box)
        candidates, scores = segmenter(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), seed_points(model, entry))
        mask = drop_specks(choose_mask(candidates, scores, region))
        reason = check_mask(mask, region)
        report["kept"].pop(entry.name, None)
        report["rejected"].pop(entry.name, None)
        if reason:
            report["rejected"][entry.name] = reason
            out.unlink(missing_ok=True)
            continue
        cv2.imwrite(str(out), mask.astype(np.uint8) * 255)
        report["kept"][entry.name] = round(float(mask.mean()), 4)
    (masks_dir / REPORT).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
