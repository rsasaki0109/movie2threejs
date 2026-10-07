"""Small, flat color patches for unseen support surfaces; no invented room texture."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def surface_patch(footprint: np.ndarray, support_y: float, background: np.ndarray,
                  rgb: np.ndarray, ring: float = 0.15) -> dict | None:
    """Fill only an object's footprint using the median color of nearby surface splats.

    Points are in aligned meters; colors are sRGB in [0, 1]. A missing color sample
    leaves the patch absent rather than guessing a color from a different surface.
    """
    xz = np.unique(footprint[:, [0, 2]], axis=0)
    if len(xz) < 3:
        return None
    try:
        hull = ConvexHull(xz)
    except QhullError:
        return None
    lo, hi = xz.min(axis=0), xz.max(axis=0)
    bxz = background[:, [0, 2]]
    outer = np.all((bxz >= lo - ring) & (bxz <= hi + ring), axis=1)
    inner = np.all((bxz > lo) & (bxz < hi), axis=1)
    same_surface = np.abs(background[:, 1] - support_y) <= 0.025
    chosen = outer & ~inner & same_surface & np.isfinite(rgb).all(axis=1)
    sample = rgb[chosen]
    if len(sample) < 3:
        return None
    polygon = xz[hull.vertices]
    patch = {
        "polygon_xz": np.round(polygon, 5).tolist(),
        "y": round(float(support_y + 0.002), 5),
        "color": np.round(np.median(sample, axis=0).astype(float).clip(0, 1), 5).tolist(),
    }
    # A tabletop or floor may change color across the footprint. Use only the
    # observed surrounding support surface, interpolating nearby median colors.
    # This remains a flat approximation, not photographic texture completion.
    if len(sample) >= 12:
        points = bxz[chosen]
        colors = []
        for vertex in polygon:
            distance = np.linalg.norm(points - vertex, axis=1)
            neighbors = np.argsort(distance)[:min(24, len(sample))]
            colors.append(np.median(sample[neighbors], axis=0).clip(0, 1))
        patch['vertex_colors'] = np.round(colors, 5).tolist()
    return patch
