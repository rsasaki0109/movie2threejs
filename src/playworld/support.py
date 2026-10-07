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
    sample = rgb[outer & ~inner & same_surface]
    sample = sample[np.isfinite(sample).all(axis=1)]
    if len(sample) < 3:
        return None
    return {
        "polygon_xz": np.round(xz[hull.vertices], 5).tolist(),
        "y": round(float(support_y + 0.002), 5),
        "color": np.round(np.median(sample, axis=0).astype(float).clip(0, 1), 5).tolist(),
    }
