"""Read/write the standard 3D Gaussian Splatting PLY layout (binary little-endian float32)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814


def read_ply(path: Path) -> dict[str, np.ndarray]:
    """Returns {property_name: (N,) float32} in file order."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        n, names = 0, []
        in_vertex = False
        while (line := f.readline().strip()) != b"end_header":
            parts = line.split()
            if parts[0] == b"format" and parts[1] != b"binary_little_endian":
                raise ValueError("only binary_little_endian PLY is supported")
            if parts[0] == b"element":
                in_vertex = parts[1] == b"vertex"
                if in_vertex:
                    n = int(parts[2])
            elif parts[0] == b"property" and in_vertex:
                if parts[1] != b"float":
                    raise ValueError(f"unsupported property type {parts[1]!r}")
                names.append(parts[2].decode())
        data = np.frombuffer(f.read(4 * n * len(names)), dtype="<f4").reshape(n, len(names))
    return {name: data[:, i].copy() for i, name in enumerate(names)}


def write_ply(path: Path, props: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(props)
    n = len(props[names[0]])
    header = "ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
    header += "".join(f"property float {name}\n" for name in names)
    header += "end_header\n"
    data = np.stack([np.asarray(props[k], dtype="<f4") for k in names], axis=1)
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def subset(props: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in props.items()}


def means(props: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([props["x"], props["y"], props["z"]], axis=1).astype(np.float64)


def opacity(props: dict[str, np.ndarray]) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-props["opacity"].astype(np.float64)))


def max_scale(props: dict[str, np.ndarray]) -> np.ndarray:
    return np.exp(np.stack([props[f"scale_{i}"] for i in range(3)], axis=1)).max(axis=1)


def make_splats(xyz: np.ndarray, rgb: np.ndarray, scale: float | np.ndarray, alpha: float = 0.95) -> dict[str, np.ndarray]:
    """Isotropic, view-independent gaussians (used for synthetic scenes and tests)."""
    n = len(xyz)
    scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), (n,))
    props = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    for i in range(3):
        props[f"f_dc_{i}"] = (rgb[:, i] - 0.5) / SH_C0
    props["opacity"] = np.full(n, np.log(alpha / (1 - alpha)))
    for i in range(3):
        props[f"scale_{i}"] = np.log(scale)
    props["rot_0"] = np.ones(n)
    for i in range(1, 4):
        props[f"rot_{i}"] = np.zeros(n)
    return props
