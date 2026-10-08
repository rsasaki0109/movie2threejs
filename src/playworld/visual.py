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
