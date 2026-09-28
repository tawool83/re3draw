"""Printable marker mat: ChArUco board + FRONT arrow + 100 mm scale bar."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .boards import BoardSpec


def _px(mm: float, dpi: int) -> int:
    return round(mm * dpi / 25.4)


def render_page(spec: BoardSpec, dpi: int = 300) -> np.ndarray:
    """Render the full page as a grayscale uint8 image (page pixel (0,0) = top-left)."""
    pw, ph = spec.page_mm
    page = Image.new("L", (_px(pw, dpi), _px(ph, dpi)), 255)

    bw, bh = spec.board_mm
    ox, oy = spec.board_origin_mm
    x0, y0, x1, y1 = _px(ox, dpi), _px(oy, dpi), _px(ox + bw, dpi), _px(oy + bh, dpi)
    board_img = spec.board.generateImage((x1 - x0, y1 - y0), marginSize=0, borderBits=1)
    page.paste(Image.fromarray(board_img), (x0, y0))

    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=_px(4.0, dpi))
    small = ImageFont.load_default(size=_px(3.0, dpi))
    margin_top, margin_bottom = oy, ph - (oy + bh)

    # Header (top margin).
    draw.text(
        (_px(ox, dpi), _px(margin_top / 2, dpi)),
        f"re3draw marker mat · {spec.name.upper()} · print at 100% / actual size",
        fill=0, font=small, anchor="lm",
    )

    # FRONT arrow (bottom margin, pointing into the mat): shoot the first photo from here.
    cx, cy = _px(pw / 2, dpi), _px(oy + bh + margin_bottom / 2, dpi)
    a = _px(min(margin_bottom * 0.3, 5.0), dpi)
    draw.polygon([(cx, cy - a), (cx - a, cy + a // 2), (cx + a, cy + a // 2)], fill=0)
    draw.text((cx + a * 2, cy), "FRONT  (take the first photo from this side)", fill=0, font=small, anchor="lm")

    # 100 mm scale bar (bottom-left) for verifying print scale.
    sx0, sy = _px(ox, dpi), cy
    sx1 = sx0 + _px(100.0, dpi)
    draw.rectangle([sx0, sy - _px(0.6, dpi), sx1, sy + _px(0.6, dpi)], fill=0)
    for x in (sx0, sx1):
        draw.rectangle([x - _px(0.3, dpi), sy - _px(2, dpi), x + _px(0.3, dpi), sy + _px(2, dpi)], fill=0)
    draw.text(((sx0 + sx1) // 2, sy - _px(2.5, dpi)), "100 mm", fill=0, font=small, anchor="mb")

    # Board id (bottom-right) so a photo of the mat tells which spec to use.
    draw.text((_px(ox + bw, dpi), cy), f"board={spec.name}", fill=0, font=font, anchor="rm")
    return np.asarray(page)


def save_pdf(spec: BoardSpec, path: str | Path, dpi: int = 300) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_page(spec, dpi)).save(path, "PDF", resolution=float(dpi))
    return path
