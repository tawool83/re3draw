"""Printed marker-mat (ChArUco) specifications and the world coordinate frame.

World frame (shared by every re3draw component):
  * origin at the centre of the printed page (= centre of the board),
  * +X toward the right edge of the page, +Y toward the top edge,
  * +Z up, out of the paper. Units are metres.
The "FRONT" edge is the bottom edge of the page, i.e. the front camera sits on -Y.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardSpec:
    name: str
    squares_x: int
    squares_y: int
    square_mm: float
    marker_mm: float
    page_mm: tuple[float, float]  # (width, height), landscape
    dictionary: int = cv2.aruco.DICT_5X5_250

    @property
    def board_mm(self) -> tuple[float, float]:
        return self.squares_x * self.square_mm, self.squares_y * self.square_mm

    @property
    def board_origin_mm(self) -> tuple[float, float]:
        """Top-left corner of the board on the page (board is centred)."""
        (pw, ph), (bw, bh) = self.page_mm, self.board_mm
        return (pw - bw) / 2, (ph - bh) / 2

    @cached_property
    def board(self) -> cv2.aruco.CharucoBoard:
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_mm / 1000.0,
            self.marker_mm / 1000.0,
            cv2.aruco.getPredefinedDictionary(self.dictionary),
        )

    @cached_property
    def corner_world_points(self) -> np.ndarray:
        """(N, 3) world coordinates of the ChArUco inner corners, indexed by corner id."""
        return board_to_world(self, self.board.getChessboardCorners())

    def page_mm_to_world(self, xy_mm: np.ndarray) -> np.ndarray:
        """Page coordinates in mm (x right, y down from top-left) -> world XY in metres."""
        xy_mm = np.asarray(xy_mm, dtype=np.float64)
        pw, ph = self.page_mm
        return np.stack([(xy_mm[..., 0] - pw / 2) / 1000.0, (ph / 2 - xy_mm[..., 1]) / 1000.0], axis=-1)


def board_to_world(spec: BoardSpec, pts: np.ndarray) -> np.ndarray:
    """OpenCV board coordinates (x right, y down, metres, z=0) -> world coordinates."""
    ox, oy = spec.board_origin_mm
    page_mm = np.asarray(pts, dtype=np.float64)[:, :2] * 1000.0 + (ox, oy)
    xy = spec.page_mm_to_world(page_mm)
    return np.hstack([xy, np.zeros((len(xy), 1))])


# Fewer, larger squares survive the grazing angles of the low capture ring better
# than many small ones; see tests/test_pose_synthetic.py.
BOARDS: dict[str, BoardSpec] = {
    "a4": BoardSpec("a4", squares_x=9, squares_y=6, square_mm=28.0, marker_mm=21.0, page_mm=(297.0, 210.0)),
    "a3": BoardSpec("a3", squares_x=12, squares_y=8, square_mm=30.0, marker_mm=22.5, page_mm=(420.0, 297.0)),
}


def get_board(name: str) -> BoardSpec:
    try:
        return BOARDS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown board {name!r}; choose from {sorted(BOARDS)}") from None
