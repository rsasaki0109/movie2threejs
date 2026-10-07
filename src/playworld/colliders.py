"""Static collision geometry from a world-frame point cloud.

Occupied voxels are merged greedily into a small number of axis-aligned boxes, which physics
engines (Rapier in the viewer) handle far better than thousands of tiny cubes or a noisy mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Box:
    center: np.ndarray
    half_extents: np.ndarray

    def to_json(self) -> dict:
        return {
            "type": "box",
            "center": [round(float(v), 4) for v in self.center],
            "half_extents": [round(float(v), 4) for v in self.half_extents],
        }


def greedy_boxes(grid: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cover occupied cells with axis-aligned boxes; returns (min_index, max_index_exclusive) pairs."""
    todo = grid.copy()
    nx, ny, nz = grid.shape
    boxes = []
    for i, j, k in zip(*np.nonzero(grid)):
        if not todo[i, j, k]:
            continue
        # grow along x
        i1 = i + 1
        while i1 < nx and todo[i1, j, k]:
            i1 += 1
        # grow along z while the whole x-run is free
        k1 = k + 1
        while k1 < nz and todo[i:i1, j, k1].all():
            k1 += 1
        # grow along y while the whole x-z slab is free
        j1 = j + 1
        while j1 < ny and todo[i:i1, j1, k:k1].all():
            j1 += 1
        todo[i:i1, j:j1, k:k1] = False
        boxes.append((np.array([i, j, k]), np.array([i1, j1, k1])))
    return boxes


def static_boxes(
    points: np.ndarray,
    voxel: float = 0.1,
    min_points: int = 2,
    floor_clearance: float = 0.05,
    close_holes: bool = True,
    min_half: float = 0.03,
    max_cells: int = 8_000_000,
) -> list[Box]:
    """Boxes for walls and furniture. Points at floor height are skipped (the floor is separate).

    Small holes (sparse texture-less walls) are closed so the player cannot slip through, and each
    box is shrunk to the points it actually contains so tables are not raised to the next voxel.
    """
    from scipy import ndimage

    pts = points[points[:, 1] > floor_clearance]
    if len(pts) == 0:
        return []
    extent = pts.max(axis=0) - pts.min(axis=0)
    while np.prod(np.floor(extent / voxel) + 3) > max_cells:
        voxel *= 1.5  # large outdoor scenes: coarsen instead of running out of memory
    origin = pts.min(axis=0) - voxel  # one empty cell of padding so closing works at the border
    idx = np.floor((pts - origin) / voxel).astype(np.int64)
    dims = idx.max(axis=0) + 2
    counts = np.bincount(np.ravel_multi_index(idx.T, dims), minlength=int(np.prod(dims))).reshape(dims)
    grid = counts >= min_points
    if close_holes:
        grid = ndimage.binary_closing(grid, structure=np.ones((3, 3, 3), bool)) | grid

    owner = np.full(grid.shape, -1, dtype=np.int64)
    cells = greedy_boxes(grid)
    for b, (lo, hi) in enumerate(cells):
        owner[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = b
    pid = owner[idx[:, 0], idx[:, 1], idx[:, 2]]
    ok = pid >= 0
    pmin = np.full((len(cells), 3), np.inf)
    pmax = np.full((len(cells), 3), -np.inf)
    np.minimum.at(pmin, pid[ok], pts[ok])
    np.maximum.at(pmax, pid[ok], pts[ok])

    boxes = []
    for b, (lo, hi) in enumerate(cells):
        vlo, vhi = origin + lo * voxel, origin + hi * voxel
        if np.isfinite(pmin[b, 0]):
            # tight in y (heights matter for standing objects), voxel extent sideways (keeps walls closed)
            # thin slabs (table tops) grow downward only, so the top surface stays where it is
            vlo[1], vhi[1] = min(pmin[b, 1], pmax[b, 1] - 2 * min_half), pmax[b, 1]
        c, h = (vlo + vhi) / 2, np.maximum((vhi - vlo) / 2, min_half)
        boxes.append(Box(c, h))
    return boxes


def drop_overlapping(boxes: list[Box], lo: np.ndarray, hi: np.ndarray, max_fraction: float = 0.3) -> list[Box]:
    """Remove boxes that are mostly inside the region [lo, hi] (leftovers around a movable object)."""
    keep = []
    for b in boxes:
        blo, bhi = b.center - b.half_extents, b.center + b.half_extents
        inter = np.clip(np.minimum(bhi, hi) - np.maximum(blo, lo), 0, None).prod()
        if inter <= max_fraction * np.prod(bhi - blo):
            keep.append(b)
    return keep


def floor_box(points: np.ndarray, margin: float = 2.0, thickness: float = 0.5) -> Box:
    lo = np.percentile(points[:, [0, 2]], 1, axis=0) - margin
    hi = np.percentile(points[:, [0, 2]], 99, axis=0) + margin
    c = (lo + hi) / 2
    h = (hi - lo) / 2
    return Box(np.array([c[0], -thickness / 2, c[1]]), np.array([h[0], thickness / 2, h[1]]))
