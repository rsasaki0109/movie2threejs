import numpy as np
from playworld.objects import movable_splat_mask


def test_movable_splats_reject_background_spill_and_giant_gaussians():
    rng = np.random.default_rng(2)
    points = rng.uniform([-.2, .8, -.2], [.2, 1.2, .2], (1000, 3))
    points = np.vstack([points, [[5, 1, 4], [0, 1, 0], [0, 1, 0]]])
    sizes = np.r_[np.full(1000, .01), .01, .4, np.nan]
    keep = movable_splat_mask(points, sizes)
    assert keep[:1000].all()
    assert not keep[1000:].any()
