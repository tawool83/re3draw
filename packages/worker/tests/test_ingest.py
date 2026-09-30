"""Photo normalisation: size, metadata and orientation of what enters the pipeline."""

import numpy as np
from PIL import Image

from re3draw_worker.ingest import JPEG_QUALITY, MAX_LONG_SIDE, ingest

GPS_IFD = 0x8825
ORIENTATION = 0x0112


def phone_photo(path, size=(4032, 3024), orientation=1, seed=0):
    """A noisy 12 MP JPEG with EXIF: GPS position and a rotation flag, as phones write them."""
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[ORIENTATION] = orientation
    exif[GPS_IFD] = {1: "N", 2: (37.0, 33.0, 0.0), 3: "E", 4: (126.0, 58.0, 0.0)}
    img.save(path, "JPEG", quality=95, exif=exif.tobytes())
    return path


def test_big_photos_are_shrunk_and_stripped(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(3):
        phone_photo(raw / f"IMG_{i}.JPG", seed=i)
    report = ingest(raw, tmp_path / "images")

    assert sorted(report.written) == ["IMG_0.jpg", "IMG_1.jpg", "IMG_2.jpg"]
    assert report.size_out == (MAX_LONG_SIDE, 1536)
    assert report.bytes_out < report.bytes_in / 3
    with Image.open(tmp_path / "images" / "IMG_0.jpg") as out:
        assert out.size == (MAX_LONG_SIDE, 1536)
        assert len(out.getexif()) == 0, "EXIF (GPS included) must not survive"


def test_rotation_flag_is_ignored_so_portrait_and_landscape_share_one_camera(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    phone_photo(raw / "landscape.jpg", orientation=1)
    phone_photo(raw / "portrait.jpg", orientation=6)  # "rotate 90 CW to display": held upright
    ingest(raw, tmp_path / "images")
    sizes = {Image.open(p).size for p in (tmp_path / "images").iterdir()}
    assert sizes == {(MAX_LONG_SIDE, 1536)}


def test_small_photos_are_not_enlarged(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    phone_photo(raw / "small.jpg", size=(800, 600))
    report = ingest(raw, tmp_path / "images")
    assert report.size_out == (800, 600)


def test_heic_and_junk_are_reported_not_fatal(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "IMG_9.HEIC").write_bytes(b"not decodable here")
    (raw / "broken.jpg").write_bytes(b"\xff\xd8 truncated")
    (raw / "notes.txt").write_text("ignored")
    phone_photo(raw / "ok.jpg", size=(1000, 750))
    report = ingest(raw, tmp_path / "images")
    assert report.written == ["ok.jpg"]
    assert report.skipped == {"IMG_9.HEIC": "heic_unsupported", "broken.jpg": "unreadable"}
    assert JPEG_QUALITY == 90
