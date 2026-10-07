"""Minimal reader/writer for COLMAP binary sparse models (cameras/images/points3D.bin)."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# model_id -> (name, number of params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
MODEL_IDS = {name: (mid, n) for mid, (name, n) in CAMERA_MODELS.items()}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def K(self) -> np.ndarray:
        """Pinhole intrinsics; distortion terms are ignored."""
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        else:
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # (w, x, y, z), world-to-camera rotation
    tvec: np.ndarray  # world-to-camera translation
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))

    @property
    def R(self) -> np.ndarray:
        return qvec_to_rotmat(self.qvec)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.tvec


@dataclass
class Reconstruction:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    xyz: np.ndarray  # (N, 3)
    rgb: np.ndarray  # (N, 3) uint8

    def sorted_images(self) -> list[Image]:
        return sorted(self.images.values(), key=lambda im: im.name)


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q if q[0] >= 0 else -q


def _read(f, fmt: str):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras_bin(path: Path) -> dict[int, Camera]:
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cid, mid, w, h = _read(f, "iiQQ")
            name, nparams = CAMERA_MODELS[mid]
            params = np.array(_read(f, "d" * nparams))
            cams[cid] = Camera(cid, name, int(w), int(h), params)
    return cams


def read_images_bin(path: Path) -> dict[int, Image]:
    images = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _read(f, "idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (npts,) = _read(f, "Q")
            raw = np.frombuffer(f.read(24 * npts), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            images[iid] = Image(
                iid,
                np.array([qw, qx, qy, qz]),
                np.array([tx, ty, tz]),
                cid,
                name.decode(),
                np.stack([raw["x"], raw["y"]], axis=1),
                raw["id"].copy(),
            )
    return images


def read_points3D_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            _pid, x, y, z, r, g, b, _err, track_len = _read(f, "QdddBBBdQ")
            f.seek(8 * track_len, 1)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    return np.array(xyz, dtype=np.float64).reshape(-1, 3), np.array(rgb, dtype=np.uint8).reshape(-1, 3)


def read_model(sparse_dir: Path) -> Reconstruction:
    sparse_dir = Path(sparse_dir)
    if not (sparse_dir / "cameras.bin").exists() and (sparse_dir / "0" / "cameras.bin").exists():
        sparse_dir = sparse_dir / "0"
    xyz, rgb = read_points3D_bin(sparse_dir / "points3D.bin")
    return Reconstruction(
        read_cameras_bin(sparse_dir / "cameras.bin"),
        read_images_bin(sparse_dir / "images.bin"),
        xyz,
        rgb,
    )


def write_model(rec: Reconstruction, sparse_dir: Path) -> None:
    """Writes a model without 2D-3D correspondences (enough for tests and gsplat init)."""
    sparse_dir = Path(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    with open(sparse_dir / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(rec.cameras)))
        for c in rec.cameras.values():
            mid, nparams = MODEL_IDS[c.model]
            f.write(struct.pack("<iiQQ", c.id, mid, c.width, c.height))
            f.write(struct.pack("<" + "d" * nparams, *c.params))
    with open(sparse_dir / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(rec.images)))
        for im in rec.images.values():
            f.write(struct.pack("<idddddddi", im.id, *im.qvec, *im.tvec, im.camera_id))
            f.write(im.name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))
    with open(sparse_dir / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(rec.xyz)))
        for i, (p, c) in enumerate(zip(rec.xyz, rec.rgb)):
            f.write(struct.pack("<QdddBBBdQ", i + 1, *p, *c, 0.0, 0))
