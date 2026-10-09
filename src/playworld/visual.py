"""Reviewed rendering corrections; collision bodies remain independent."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from . import objects


def object_surface_mask(points: np.ndarray, world_points: np.ndarray,
                        sizes: np.ndarray, views: list[objects.View],
                        masks: list[np.ndarray], label: int, support_y: float,
                        background_ratio: float = .8, max_size: float = .08,
                        visibility_points: np.ndarray | None = None) -> np.ndarray:
    """Remove demonstrable mask spill and broad/below-support splats conservatively.

    Applied to a reviewed object's existing splats, not to the whole room. Rejected
    points are removed rather than becoming a static copy of the movable object.
    The room supplies occlusion evidence. Unseen surfaces are retained: absence of
    a positive mask vote is not evidence that a surface belongs to the background.
    Two visible background votes and a strong majority are required for rejection.
    The support tolerance retains contact surfaces while rejecting tabletop spill.
    """
    reference = points if visibility_points is None else np.vstack([points, visibility_points])
    seen = np.zeros(len(points), dtype=int)
    outside = np.zeros(len(points), dtype=int)
    for view, mask in zip(views, masks):
        visible, u, v = objects.visible(reference, view)
        idx = np.flatnonzero(visible[:len(points)])
        seen[idx] += 1
        outside[idx] += mask[v[idx], u[idx]] != label
    spill = (outside >= 2) & (outside >= background_ratio * np.maximum(seen, 1))
    return (~spill & np.isfinite(sizes) & (sizes <= max_size)
            & (world_points[:, 1] >= support_y - .005))


def bottom_cap(points: np.ndarray, support_y: float, centroid: np.ndarray,
               base_band: float = .12, inset: float = .005) -> dict | None:
    """Inset horizontal bottom for a manually reviewed open box, in body coordinates.

    Fit a small oriented rectangle to the photographed lower sides. This deliberately
    does not close the top or build sloped surfaces through protruding contents.
    It is an approximate unseen bottom, not recovered photographic geometry.
    """
    base = points[(points[:, 1] >= support_y - .005)
                  & (points[:, 1] <= support_y + base_band), :][:, [0, 2]]
    if len(base) < 20:
        return None
    try:
        hull = ConvexHull(base)
    except QhullError:
        return None
    edges = np.roll(base[hull.vertices], -1, axis=0) - base[hull.vertices]
    angles = np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2), 6))
    best = None
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        axes = np.array([[c, -s], [s, c]])
        local = base @ axes
        lo, hi = np.percentile(local, [2, 98], axis=0)
        width = hi - lo
        if width.min() <= 2 * inset:
            continue
        area = np.prod(width)
        if best is None or area < best[0]:
            best = area, axes, lo + inset, hi - inset
    if best is None:
        return None
    _, axes, lo, hi = best
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                        [hi[0], hi[1]], [lo[0], hi[1]]]) @ axes.T
    vertices = np.column_stack([corners[:, 0], np.full(4, support_y + .007), corners[:, 1]])
    return {"kind": "bottom_cap", "vertices": np.round(vertices - centroid, 5).tolist(),
            "approximate": True}


def clipped_hull_fill(points: np.ndarray, min_y: float, centroid: np.ndarray) -> dict | None:
    """Approximate only the upper interior of a reviewed object's convex body.

    The explicit world-height cut leaves chair leg gaps open. Intersect hull
    edges with the cut plane, rather than dropping vertices and changing the
    upper outline. This is visual geometry; never use it for physics.
    """
    points = np.asarray(points, dtype=float)
    centroid = np.asarray(centroid, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or centroid.shape != (3,):
        raise ValueError('Expected points (N, 3) and a three-coordinate centroid')
    if not np.isfinite(points).all() or not np.isfinite(centroid).all() or not np.isfinite(min_y):
        raise ValueError('Visual hull coordinates and cut height must be finite')
    if len(points) < 4:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    kept = list(points[hull.vertices][points[hull.vertices, 1] >= min_y])
    edges = {tuple(sorted((int(a), int(b)))) for face in hull.simplices
             for a, b in zip(face, np.roll(face, -1))}
    for a, b in sorted(edges):
        start, end = points[a], points[b]
        if (start[1] < min_y < end[1]) or (end[1] < min_y < start[1]):
            kept.append(start + (end-start) * ((min_y-start[1]) / (end[1]-start[1])))
    if len(kept) < 4:
        return None
    kept = np.unique(np.round(kept, 10), axis=0)
    try:
        clipped = ConvexHull(kept)
    except QhullError:
        return None
    if clipped.volume <= 1e-10:
        return None
    return {'kind': 'convex_hull', 'vertices': np.round(kept[clipped.vertices]-centroid, 5).tolist(),
            'min_world_y': float(min_y), 'approximate': True}
