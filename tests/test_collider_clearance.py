import numpy as np
import pytest
from playworld import colliders


def test_carving_preserves_merged_table_without_body_overlap():
    original = colliders.Box(np.array([0., .9, 0.]), np.array([2., .2, 1.]))
    lo, hi = np.array([-.2, .8, -.2]), np.array([.2, 1.4, .2])
    pieces = colliders.carve_boxes([original], lo, hi)
    removed = np.prod(np.minimum(original.center + original.half_extents, hi) - np.maximum(original.center - original.half_extents, lo))
    assert sum(np.prod(2 * p.half_extents) for p in pieces) == pytest.approx(np.prod(2 * original.half_extents) - removed)
    for p in pieces:
        overlap = np.minimum(p.center + p.half_extents, hi) - np.maximum(p.center - p.half_extents, lo)
        assert np.any(overlap <= 1e-12)
    assert any(np.all(np.abs(np.array([1., .9, 0.]) - p.center) <= p.half_extents) for p in pieces)


def test_support_slab_stays_below_object():
    footprint = np.array([[-.2, .905, -.2], [.2, 1.2, .2]])
    box = colliders.support_box(footprint, .9)
    assert box.center[1] + box.half_extents[1] == pytest.approx(.9)
    assert box.center[1] + box.half_extents[1] < footprint[:, 1].min()
