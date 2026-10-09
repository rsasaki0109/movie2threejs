"""Conservative rejection of diffuse volume splats, in the aligned metric frame."""
from __future__ import annotations

import numpy as np


def surface_splat_mask(scales: np.ndarray, opacity: np.ndarray) -> np.ndarray:
    """Keep thin surfaces, including broad walls; reject thick translucent blobs.

    ``scales`` are Gaussian standard deviations in assumed meters, not logs.
    Low opacity alone is not evidence of noise: blanket alpha pruning punches
    holes in surfaces built from overlapping translucent Gaussians.
    """
    scales = np.asarray(scales, dtype=float)
    opacity = np.asarray(opacity, dtype=float)
    if scales.ndim != 2 or scales.shape[1] != 3 or opacity.shape != (len(scales),):
        raise ValueError("expected scales (N, 3) and opacity (N,)")
    valid = (np.isfinite(scales).all(axis=1) & (scales > 0).all(axis=1)
             & np.isfinite(opacity) & (opacity >= 0) & (opacity <= 1))
    thick, broad = scales.min(axis=1), scales.max(axis=1)
    diffuse = ((thick > .025) & (broad > .1) & (opacity < .5))
    diffuse |= (thick > .08) | ((broad > .5) & (opacity < .7))
    return valid & ~diffuse


def floor_footprint(reference, band=.06, cell=.2):
    """Largest connected observed floor component, in an aligned y-up world.

    Select reference points near y=0, then connect adjacent horizontal voxels
    (eight neighbors). Weight components by original point count so an isolated
    exterior patch cannot expand the room footprint. This is a heuristic, not
    a room boundary or a free-space measurement. Assumed scale affects it.
    """
    from scipy.spatial import ConvexHull, QhullError, cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    reference = np.asarray(reference, dtype=float)
    if (reference.ndim != 2 or reference.shape[1] != 3
            or not np.isfinite(reference).all()):
        raise ValueError('reference must be a finite Nx3 array')
    if not np.isfinite([band, cell]).all() or band <= 0 or cell <= 0:
        raise ValueError('floor band and voxel size must be finite and positive')
    floor = reference[np.abs(reference[:, 1]) < band]
    if len(floor) < 3:
        raise ValueError('at least three observed floor points are required')
    voxels, inverse = np.unique(np.floor(floor[:, [0, 2]]/cell), axis=0, return_inverse=True)
    pairs = cKDTree(voxels).query_pairs(np.sqrt(2)+1e-5, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(voxels), len(voxels)))
    _, labels = connected_components(graph, directed=False)
    weights = np.bincount(labels[inverse])
    selected = floor[labels[inverse] == weights.argmax()]
    if len(selected) < 3:
        raise ValueError('observed floor footprint must have nonzero horizontal area')
    try:
        ConvexHull(selected[:, [0, 2]])
    except QhullError as error:
        raise ValueError('observed floor footprint must have nonzero horizontal area') from error
    return selected


def interior_support_mask(points, reference, cameras, max_distance=.3,
                          margin=.05, min_height=.25, max_height=2.75):
    """Experimental support filter in an aligned, y-up world (assumed meters).

    Reject unsupported centers only inside the horizontal anchor footprint,
    inset by ``margin``, and the specified height band. This is not measured
    free space: untracked interior surfaces can be removed. Exterior geometry,
    the footprint boundary, floor and ceiling outside the band are preserved.
    Inputs must already share the same world alignment and scale. Anchors are
    camera centers by default; ``floor_footprint`` provides an opt-in alternative.
    """
    from scipy.spatial import ConvexHull, QhullError, cKDTree

    arrays = [np.asarray(a, dtype=float) for a in (points, reference, cameras)]
    for a in arrays:
        if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
            raise ValueError('points, reference and cameras must be finite Nx3 arrays')
    points, reference, cameras = arrays
    settings = [max_distance, margin, min_height, max_height]
    if (not np.isfinite(settings).all() or max_distance <= 0 or margin < 0
            or min_height >= max_height):
        raise ValueError('invalid interior support distance, margin or height range')
    if not len(reference) or len(cameras) < 3:
        raise ValueError('reference points and at least three camera centers are required')
    try:
        hull = ConvexHull(cameras[:, [0, 2]])
    except QhullError as error:
        raise ValueError('camera footprint must have nonzero horizontal area') from error
    inside = (points[:, 1] > min_height) & (points[:, 1] < max_height)
    for a, b, c in hull.equations:
        inside &= a*points[:, 0] + b*points[:, 2] + c < -margin
    keep = np.ones(len(points), dtype=bool)
    ids = np.flatnonzero(inside)
    if len(ids):
        distance, _ = cKDTree(reference).query(points[ids], workers=4)
        keep[ids] = distance <= max_distance
    return keep
