"""The ray-traced mug used by the end-to-end benchmark (modal_app.py::mug_benchmark)."""

import cv2
import numpy as np
import pytest

from re3draw_worker.boards import get_board
from re3draw_worker.synthetic import Mug, Renderer, look_at, orbit_position

W, H = 480, 360


@pytest.fixture(scope="module")
def renderer():
    K = np.array([[420.0, 0, W / 2], [0, 420.0, H / 2], [0, 0, 1]])
    return Renderer(get_board("a3"), K, np.zeros(4), (W, H)), K


def _view(az_deg, el_deg, dist=0.35):
    return look_at(orbit_position(np.radians(az_deg), np.radians(el_deg), dist), np.array([0.01, 0.0, 0.05]))


def _pixel(K, R, t, X):
    uv, _ = cv2.projectPoints(np.asarray([X], float), cv2.Rodrigues(R)[0], t, K, np.zeros(4))
    return np.round(uv.reshape(2)).astype(int)


def test_the_mat_shows_through_the_handle_loop(renderer):
    r, K = renderer
    mug = Mug()
    # Seen from the front the handle sticks out to the right (+X); the loop's hole sits between the
    # wall and the handle's outer arc, at the handle's height.
    hole = np.array([mug.radius + 0.012, 0.0, mug.handle_height])
    tube = np.array([mug.radius - 0.004 + mug.handle_radius, 0.0, mug.handle_height])
    R, t = _view(0, 0.5)
    _, mask = r.render_mug(R, t, None, mug)
    x, y = _pixel(K, R, t, tube)
    assert mask[y, x], "outer arc of the handle"
    x, y = _pixel(K, R, t, hole)
    assert not mask[y, x], "the loop must be open: background visible inside it"


def test_front_logo_faces_minus_y_and_handle_is_on_plus_x(renderer):
    r, K = renderer
    mug = Mug()
    R, t = _view(0, 20)  # front
    front, _ = r.render_mug(R, t, None, mug)
    x, y = _pixel(K, R, t, [0.0, -mug.radius, mug.height / 2])
    assert front[y, x].max() < 60, "dark logo in the middle of the front"
    R, t = _view(180, 20)  # back: no logo there
    back, _ = r.render_mug(R, t, None, mug)
    x, y = _pixel(K, R, t, [0.0, mug.radius, mug.height / 2])
    assert back[y, x].max() > 60
    R, t = _view(90, 20)  # from +X the handle is in front of the wall
    _, mask = r.render_mug(R, t, None, mug)
    x, y = _pixel(K, R, t, mug.handle_centres[len(mug.handle_centres) // 2])
    assert mask[y, x]


def test_object_mask_matches_the_photo(renderer):
    r, _ = renderer
    R, t = _view(30, 40)
    img, mask = r.render_mug(R, t, None, Mug())
    assert 0.02 < mask.mean() < 0.4
    # Off the mug the photo is the grey mat/table (equal channels); on it, the mug's colours.
    grey = np.ptp(img.astype(int), axis=2) < 3
    assert grey[~mask].mean() > 0.99
    assert (~grey[mask]).mean() > 0.3
