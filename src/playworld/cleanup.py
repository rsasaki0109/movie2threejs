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
