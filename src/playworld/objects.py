"""Lift per-frame 2D instance masks onto 3D gaussians and build rigid-body colliders.

Each gaussian center is projected into every frame. A point-splat z-buffer decides visibility,
so gaussians hidden behind an object do not inherit its label. A gaussian joins an object when
the object's label wins a majority of the frames in which the gaussian is visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull


@dataclass
class View:
    K: np.ndarray  # 3x3 intrinsics
    R: np.ndarray  # world-to-camera rotation
    t: np.ndarray  # world-to-camera translation
    width: int
    height: int


def project(points: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates (u, v) and depth z for each point."""
    pc = points @ view.R.T + view.t
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (pc @ view.K.T)[:, :2] / z[:, None]
    return uv[:, 0], uv[:, 1], z


def visible(points: np.ndarray, view: View, cell: int = 4, depth_tol: float = 0.03) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Visibility mask via a coarse point z-buffer; also returns integer pixel coordinates."""
    u, v, z = project(points, view)
    ok = (z > 1e-6) & (u >= 0) & (u < view.width) & (v >= 0) & (v < view.height)
    ui = np.zeros(len(points), dtype=np.int64)
    vi = np.zeros(len(points), dtype=np.int64)
    ui[ok] = u[ok].astype(np.int64)
    vi[ok] = v[ok].astype(np.int64)
    gw, gh = -(-view.width // cell), -(-view.height // cell)
    cell_idx = (vi // cell) * gw + (ui // cell)
    zbuf = np.full(gw * gh, np.inf)
    np.minimum.at(zbuf, cell_idx[ok], z[ok])
    vis = ok.copy()
    vis[ok] = z[ok] <= zbuf[cell_idx[ok]] * (1 + depth_tol)
    return vis, ui, vi


def lift_labels(
    points: np.ndarray,
    views: list[View],
    masks: list[np.ndarray],
    min_votes: int = 2,
    min_ratio: float = 0.5,
) -> np.ndarray:
    """Label per point (0 = background). masks[i] is an (H, W) int array, 0 = background."""
    num_labels = int(max(m.max() for m in masks)) + 1
    votes = np.zeros((len(points), num_labels), dtype=np.int32)
    seen = np.zeros(len(points), dtype=np.int32)
    for view, mask in zip(views, masks):
        vis, ui, vi = visible(points, view)
        idx = np.flatnonzero(vis)
        np.add.at(votes, (idx, mask[vi[idx], ui[idx]]), 1)
        seen[idx] += 1
    if num_labels == 1:
        return np.zeros(len(points), dtype=np.int32)
    obj = votes[:, 1:].argmax(axis=1) + 1
    best = votes[np.arange(len(points)), obj]
    keep = (best >= min_votes) & (best >= min_ratio * np.maximum(seen, 1))
    return np.where(keep, obj, 0).astype(np.int32)


def inlier_mask(points: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Drop stray gaussians far from the object's bulk (median absolute deviation per axis)."""
    med = np.median(points, axis=0)
    mad = np.median(np.abs(points - med), axis=0) + 1e-9
    return (np.abs(points - med) <= k * 1.4826 * mad).all(axis=1)


def movable_splat_mask(points: np.ndarray, sizes: np.ndarray, max_size: float = 0.1) -> np.ndarray:
    """Keep the object's bulk and reject broad foreground/background spill."""
    return inlier_mask(points) & np.isfinite(sizes) & (sizes <= max_size)


@dataclass
class RigidObject:
    label: int
    name: str
    centroid: np.ndarray
    hull: np.ndarray  # hull vertices relative to the centroid
    volume: float

    def to_json(self, splat_path: str, density: float) -> dict:
        return {
            "id": self.label,
            "name": self.name,
            "splat": splat_path,
            "centroid": [round(float(v), 4) for v in self.centroid],
            "hull": [[round(float(v), 4) for v in p] for p in self.hull],
            "mass": round(max(self.volume * density, 0.05), 3),
        }


def support_height(obj_pts: np.ndarray, static_pts: np.ndarray, cell: float = 0.05, ring: float = 0.12, band: float = 0.04, min_cover: float = 0.4) -> float:
    """Height of the surface the object rests on (floor = 0).

    The surface right under an object is never visible, so we look at a ring around its footprint
    and take the highest horizontal band (below the object) that covers most of that ring.
    """
    lo, hi = obj_pts[:, [0, 2]].min(axis=0), obj_pts[:, [0, 2]].max(axis=0)
    bottom = np.percentile(obj_pts[:, 1], 5)
    xz = static_pts[:, [0, 2]]
    in_outer = np.all((xz > lo - ring) & (xz < hi + ring), axis=1)
    in_inner = np.all((xz > lo) & (xz < hi), axis=1)
    cand = static_pts[in_outer & ~in_inner & (static_pts[:, 1] < bottom + band) & (static_pts[:, 1] > band)]
    if len(cand) == 0:
        return 0.0
    outer = np.ceil((hi - lo + 2 * ring) / cell).astype(int)
    inner = np.floor((hi - lo) / cell).astype(int).clip(0)
    ring_cells = max(int(np.prod(outer) - np.prod(inner)), 1)
    for y in np.sort(np.unique(np.round(cand[:, 1] / band)))[::-1] * band:
        p = cand[np.abs(cand[:, 1] - y) < band]
        cells = np.unique(np.floor((p[:, [0, 2]] - (lo - ring)) / cell).astype(int), axis=0)
        if len(cells) / ring_cells >= min_cover:
            return float(np.median(p[:, 1]))
    return 0.0


def rigid_object(points_world: np.ndarray, label: int, name: str, support_y: float = 0.0) -> RigidObject | None:
    pts = points_world[inlier_mask(points_world)]
    if len(pts) < 8:
        return None
    # A few noisy points can lie below an otherwise well-supported tabletop.
    # Keep the collision body above its support rather than starting embedded.
    pts = pts.copy()
    pts[:, 1] = np.maximum(pts[:, 1], support_y + 0.005)
    # Only the visible surfaces were reconstructed. Assume the object is solid down to the surface
    # it rests on, so it has real volume and stands instead of being a hollow shell.
    foot = pts.copy()
    foot[:, 1] = support_y + 0.005
    pts = np.vstack([pts, foot])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    verts = pts[hull.vertices]
    centroid = verts.mean(axis=0)
    return RigidObject(label, name, centroid, verts - centroid, float(hull.volume))


def inside_hull(points: np.ndarray, obj: RigidObject, grow: float = 0.03) -> np.ndarray:
    """Points inside the object's hull, grown by `grow` meters sideways and upward (not downward,
    so the surface it rests on stays put)."""
    from scipy.spatial import Delaunay

    rel = obj.hull.copy()
    r = np.linalg.norm(rel[:, [0, 2]], axis=1, keepdims=True) + 1e-9
    rel[:, [0, 2]] *= (r + grow) / r
    rel[rel[:, 1] > 0, 1] += grow
    return Delaunay(rel + obj.centroid).find_simplex(points) >= 0
