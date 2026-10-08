"""Compare explicitly corresponding camera models in a shared similarity frame."""
from __future__ import annotations

import numpy as np


def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit target = scale * rotation @ source + translation, without reflection."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape or len(source) < 3:
        raise ValueError('Expected corresponding (N, 3) points with N >= 3')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Camera centers must be finite')
    x, y = source - source.mean(0), target - target.mean(0)
    if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError('Camera paths must contain at least two independent directions')
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(sign) @ vt
    scale = float(singular @ sign / np.mean(np.sum(x*x, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('No positive similarity scale')
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    return scale, rotation, translation


def rotation_angle(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError('Expected a finite 3x3 rotation')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError('Expected a proper rotation')
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))))


def transform_camera(rotation, translation, scale, frame_rotation, frame_translation):
    """Transform a COLMAP world-to-camera pose to the similarity target frame."""
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    transformed_rotation = rotation @ frame_rotation.T
    transformed_translation = scale * translation - transformed_rotation @ frame_translation
    return transformed_rotation, transformed_translation
