"""Gravity, floor, and metric-scale estimation for handheld phone captures.

The reconstruction from a feed-forward model (or SfM) has an arbitrary rotation and scale.
We recover a playable world frame from two weak but reliable priors of handheld video:

1. People hold the phone roughly upright, so the average image "down" axis is close to gravity.
2. The phone is carried at roughly constant eye height above the floor.

World frame convention (matches three.js): +y up, floor at y = 0, first camera looks along -z,
units in meters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Floor:
    normal_down: np.ndarray  # unit normal pointing toward gravity
    offset: float  # plane: dot(p, normal_down) == offset
    inliers: np.ndarray  # bool mask over the input points


def down_from_cameras(rotations: np.ndarray) -> np.ndarray:
    """Average image-down direction (+y in OpenCV camera frame) expressed in world coordinates.

    rotations: (M, 3, 3) world-to-camera rotations.
    """
    downs = np.einsum("mij,i->mj", rotations, np.array([0.0, 1.0, 0.0]))  # R^T @ e_y
    d = downs.sum(axis=0)
    return d / np.linalg.norm(d)


def fit_floor(
    points: np.ndarray,
    camera_centers: np.ndarray,
    down: np.ndarray,
    max_tilt_deg: float = 25.0,
    iters: int = 1000,
    inlier_frac: float = 0.01,
    seed: int = 0,
) -> Floor:
    """RANSAC for the lowest well-supported horizontal plane below the cameras.

    inlier_frac: inlier distance threshold as a fraction of the robust scene extent.
    """
    rng = np.random.default_rng(seed)
    h = points @ down
    hc = camera_centers @ down
    extent = np.linalg.norm(np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0))
    thresh = inlier_frac * extent
    cam_to_lowest = np.percentile(h, 98) - np.median(hc)
    # The floor lies clearly below the cameras (larger h); tables and beds are higher up.
    pool = np.flatnonzero(h > np.median(hc) + 0.3 * cam_to_lowest)
    if len(pool) < 3:
        raise ValueError("not enough points below the cameras to fit a floor")
    cos_tol = np.cos(np.deg2rad(max_tilt_deg))
    pool_pts = points[pool]

    # Textured tabletops can have far more tracked points than a bare floor.
    # Sample the lower tail as well, then prefer the lowest supported plane.
    lower_pool = np.argsort(pool_pts @ down)[-max(3, int(np.ceil(len(pool_pts) * 0.2))):]
    min_support = max(3, int(np.ceil(len(pool_pts) * 0.05)))
    lowest_level = np.percentile(h, 98)
    camera_level = np.median(hc)
    best_n, best_d, best_count = None, None, -1
    lower_candidates = []
    dominant_level = None
    for iteration in range(iters * 2):
        if iteration == iters and best_n is not None:
            dominant_level = camera_level + (best_d - np.median(camera_centers @ best_n)) / (best_n @ down)
        sample = (rng.choice(len(pool_pts), 3, replace=False) if iteration < iters
                  else rng.choice(lower_pool, 3, replace=False))
        a, b, c = pool_pts[sample]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        if n @ down < 0:
            n = -n
        if n @ down < cos_tol:
            continue
        d = n @ a
        count = int(np.count_nonzero(np.abs(pool_pts @ n - d) < thresh))
        if iteration < iters:
            if count > best_count:
                best_n, best_d, best_count = n, d, count
            continue
        # Evaluate the plane below the median camera position, not the world
        # origin; this also avoids rewarding tilted planes extrapolated afar.
        level = camera_level + (d - np.median(camera_centers @ n)) / (n @ down)
        if (count < min_support or level > lowest_level + thresh or level <= camera_level
                or dominant_level is None or level <= dominant_level + 3 * thresh):
            continue
        lower_candidates.append((n, d, count, level))
    if lower_candidates:
        lowest = max(candidate[3] for candidate in lower_candidates)
        best_n, best_d, best_count, _ = max(
            (candidate for candidate in lower_candidates if candidate[3] >= lowest - 3 * thresh),
            key=lambda candidate: candidate[2],
        )
    if best_n is None:
        raise ValueError("no horizontal plane found; is the video upright?")

    # Least-squares refinement on the inliers.
    inl = np.abs(points @ best_n - best_d) < thresh
    for _ in range(2):
        p = points[inl]
        centroid = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - centroid, full_matrices=False)
        n = vt[-1]
        if n @ down < 0:
            n = -n
        d = n @ centroid
        inl = np.abs(points @ n - d) < thresh
    return Floor(n, float(d), inl)


def camera_heights(camera_centers: np.ndarray, floor: Floor) -> np.ndarray:
    """Distance of each camera above the floor (positive = above)."""
    return floor.offset - camera_centers @ floor.normal_down


def world_alignment(
    floor: Floor,
    camera_centers: np.ndarray,
    first_rotation: np.ndarray,
    eye_height: float = 1.5,
) -> np.ndarray:
    """4x4 similarity mapping reconstruction coordinates to the metric, y-up world frame."""
    up = -floor.normal_down
    forward = first_rotation.T @ np.array([0.0, 0.0, 1.0])  # first camera optical axis in world
    forward = forward - (forward @ up) * up
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError("first camera looks straight up or down")
    z = -forward / np.linalg.norm(forward)  # three.js cameras look along -z
    y = up
    x = np.cross(y, z)
    R = np.stack([x, y, z])  # rows: world axes expressed in reconstruction coordinates

    heights = camera_heights(camera_centers, floor)
    median_h = float(np.median(heights))
    if median_h <= 0:
        raise ValueError("cameras are not above the estimated floor")
    s = eye_height / median_h

    c = camera_centers.mean(axis=0)
    origin = c + (floor.offset - c @ floor.normal_down) * floor.normal_down  # projection onto floor

    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = -s * R @ origin
    return T


def apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]
