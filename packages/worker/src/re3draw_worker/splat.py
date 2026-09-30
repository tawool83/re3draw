"""Gaussian splat cloud and its file formats (.ply, .spz).

The in-memory cloud keeps the *raw* (pre-activation) parameters that training optimises, because
that is what both target formats store: opacity as a logit, scale as a natural log, colour as
spherical-harmonic coefficients. Converting to display values is the viewer's job.

`.ply` is the de-facto interchange format introduced by the original 3D Gaussian Splatting code and
is what every viewer reads. `.spz` (Niantic, MIT) is ~10x smaller and is what we ship to phones;
rather than re-implement its packed binary layout we hand our `.ply` to the reference encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Y_00, the constant SH basis function: colour = 0.5 + SH_C0 * f_dc.
SH_C0 = 0.28209479177387814


def sh_coeffs(degree: int) -> int:
    """Number of SH coefficients per colour channel for a given degree."""
    return (degree + 1) ** 2


def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """Linear RGB in [0, 1] -> degree-0 SH coefficient."""
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / SH_C0


@dataclass
class GaussianCloud:
    """N gaussians in world space, metres, with raw (pre-activation) parameters.

    means      (N, 3)          centres
    scales     (N, 3)          log of the axis lengths
    quats      (N, 4)          rotation, w x y z, not necessarily normalised
    opacities  (N,)            logit of alpha
    sh0        (N, 1, 3)       degree-0 SH (the base colour)
    shN        (N, K - 1, 3)   higher-degree SH, K = (degree + 1) ** 2
    """

    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacities: np.ndarray
    sh0: np.ndarray
    shN: np.ndarray

    def __post_init__(self) -> None:
        for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            setattr(self, name, np.ascontiguousarray(getattr(self, name), dtype=np.float32))
        n, rest = len(self.means), self.shN.shape[1] if self.shN.ndim == 3 else -1
        shapes = {"means": (n, 3), "scales": (n, 3), "quats": (n, 4),
                  "opacities": (n,), "sh0": (n, 1, 3), "shN": (n, rest, 3)}
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
        if sh_coeffs(self.sh_degree) != rest + 1:
            raise ValueError(f"shN has {rest} coefficients per channel, which is not (degree + 1)^2 - 1")

    def __len__(self) -> int:
        return len(self.means)

    @property
    def sh_degree(self) -> int:
        return int(round(np.sqrt(self.shN.shape[1] + 1))) - 1


def _ply_property_names(rest_count: int) -> list[str]:
    # Order is fixed by the original 3DGS implementation; viewers rely on the names, not the order,
    # but writing them in the canonical order keeps diffs against reference files readable.
    return [
        "x", "y", "z", "nx", "ny", "nz",
        *(f"f_dc_{i}" for i in range(3)),
        *(f"f_rest_{i}" for i in range(rest_count)),
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]


def write_ply(cloud: GaussianCloud, path: str | Path) -> Path:
    """Write the 3DGS interchange `.ply` (binary little endian, every property float32)."""
    n = len(cloud)
    # 3DGS stores the higher SH channel-major: all coefficients of R, then G, then B.
    rest = np.transpose(cloud.shN, (0, 2, 1)).reshape(n, -1)
    columns = [
        cloud.means, np.zeros((n, 3), dtype=np.float32),  # normals are unused but expected
        cloud.sh0.reshape(n, 3), rest,
        cloud.opacities.reshape(n, 1), cloud.scales, cloud.quats,
    ]
    data = np.concatenate([np.ascontiguousarray(c, dtype=np.float32) for c in columns], axis=1)
    names = _ply_property_names(rest.shape[1])
    if data.shape[1] != len(names):
        raise AssertionError(f"{data.shape[1]} columns for {len(names)} properties")

    header = "\n".join([
        "ply", "format binary_little_endian 1.0", f"element vertex {n}",
        *(f"property float {name}" for name in names), "end_header", "",
    ])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.astype("<f4", copy=False).tobytes())
    return path


def read_ply(path: str | Path) -> GaussianCloud:
    """Read a binary little-endian 3DGS `.ply` (ours, or another tool's) back into a cloud.

    Properties are matched by name, so their order and any extra ones (e.g. normals) do not matter.
    """
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header") + len(b"end_header")
    end += 2 if raw[end:end + 2] == b"\r\n" else 1
    lines = raw[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError(f"{path}: only binary little-endian PLY is supported")
    count = next(int(line.split()[2]) for line in lines if line.startswith("element vertex"))
    names, dtypes = [], []
    for line in lines:
        if line.startswith("property"):
            _, kind, name = line.split()
            names.append(name)
            dtypes.append({"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1",
                           "int": "<i4", "uint": "<u4"}[kind])
    data = np.frombuffer(raw, dtype=np.dtype(list(zip(names, dtypes))), count=count, offset=end)
    col = lambda *keys: np.stack([data[k].astype(np.float32) for k in keys], axis=1)  # noqa: E731
    rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    rest = col(*rest_names) if rest_names else np.zeros((count, 0), np.float32)
    return GaussianCloud(
        means=col("x", "y", "z"),
        scales=col("scale_0", "scale_1", "scale_2"),
        quats=col("rot_0", "rot_1", "rot_2", "rot_3"),
        opacities=data["opacity"].astype(np.float32),
        sh0=col("f_dc_0", "f_dc_1", "f_dc_2").reshape(count, 1, 3),
        shN=np.transpose(rest.reshape(count, 3, -1), (0, 2, 1)),  # channel-major on disk
    )


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    m = np.asarray(R, dtype=np.float64)
    w = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return np.array([w, x, y, z])


def transformed(cloud: GaussianCloud, R: np.ndarray, scale: float, t: np.ndarray) -> GaussianCloud:
    """The cloud moved by x -> scale * R @ x + t. Higher-order SH are not rotated, so this is exact
    for degree-0 clouds (such as generated ones) and only approximate for view-dependent colour."""
    q = rotmat_to_quat(R)
    return GaussianCloud(
        means=(scale * cloud.means @ np.asarray(R).T + np.asarray(t)).astype(np.float32),
        scales=cloud.scales + np.float32(np.log(scale)),
        quats=_quat_multiply(np.broadcast_to(q, cloud.quats.shape), cloud.quats),
        opacities=cloud.opacities, sh0=cloud.sh0, shN=cloud.shN,
    )


class SpzUnavailable(RuntimeError):
    """The optional `spz` encoder is not installed."""


def write_spz(ply_path: str | Path, path: str | Path) -> Path:
    """Convert a 3DGS `.ply` to `.spz` using Niantic's reference encoder.

    Going through the reference `.ply` reader rather than packing the bitstream ourselves means the
    quantisation and coordinate conventions are whatever the format's authors say they are.
    """
    try:
        import spz  # noqa: PLC0415  (optional dependency, installed only where we export)
    except ImportError as e:
        raise SpzUnavailable(
            "install the .spz encoder to write .spz: pip install 'git+https://github.com/nianticlabs/spz'"
        ) from e
    cloud = spz.load_splat_from_ply(str(ply_path), spz.UnpackOptions())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not spz.save_spz(cloud, spz.PackOptions(), str(path)):
        raise RuntimeError(f"spz encoder failed to write {path}")
    return path
