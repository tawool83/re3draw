"""Capture-dome coverage, with the three capture habits of the mug benchmark."""

import numpy as np
import pytest

from re3draw_worker.coverage import assess, camera_angles
from re3draw_worker.synthetic import orbit_position


def ring_centres(rings, azimuth=None, distance=0.55):
    out = []
    for elev, count in rings:
        for k in range(count):
            if azimuth is None:
                a = 360.0 * k / count
            else:
                a = azimuth[0] + (azimuth[1] - azimuth[0]) * k / max(count - 1, 1)
            out.append(orbit_position(np.radians(a), np.radians(elev), distance))
    return out


@pytest.mark.parametrize("az, el", [(0, 15), (90, 40), (180, 70), (270, 5), (359, 60)])
def test_angles_match_the_orbit_convention(az, el):
    a, e = camera_angles(orbit_position(np.radians(az), np.radians(el), 0.5))
    assert a == pytest.approx(az % 360, abs=1e-6) and e == pytest.approx(el, abs=1e-6)


def test_the_capture_guide_covers_everything():
    cov = assess(ring_centres([(15, 18), (40, 18), (70, 10)]))
    assert cov.status == "ok" and cov.empty == []
    assert cov.max_gap_deg == pytest.approx(20.0)


def test_front_only_is_refused_and_names_the_back():
    cov = assess(ring_centres([(15, 9), (40, 9), (70, 5)], azimuth=(-60, 60)))
    assert cov.status == "fail"
    assert cov.max_gap_deg == pytest.approx(240.0)
    assert cov.gap_centre == "back"
    assert "low/back" in cov.empty and "top/back" in cov.empty
    assert "back" in cov.message()


def test_few_photos_warn_with_the_empty_cells():
    cov = assess(ring_centres([(15, 4), (40, 4), (70, 4)]))
    assert cov.status == "warn"
    assert cov.max_gap_deg == pytest.approx(90.0)
    assert set(cov.empty) == {f"{ring}/{s}" for ring in ("low", "mid")
                              for s in ("front-right", "back-right", "back-left", "front-left")}
    assert cov.counts["top"] == [1, 1, 1, 1]


def test_a_missing_top_ring_is_a_warning_not_a_failure():
    cov = assess(ring_centres([(15, 18), (40, 18)]))
    assert cov.status == "warn"
    assert {e for e in cov.empty if e.startswith("top/")} == {"top/front", "top/right", "top/back", "top/left"}


def test_no_cameras_fails():
    assert assess([]).status == "fail"
