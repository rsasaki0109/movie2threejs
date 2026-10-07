import numpy as np
import pytest

from playworld.gravity import fit_floor


def test_sparse_floor_wins_over_a_dense_table_and_low_outliers():
    rng = np.random.default_rng(24)
    floor = np.column_stack([rng.uniform(-3, 3, 500), rng.normal(0, .002, 500), rng.uniform(-3, 3, 500)])
    table = np.column_stack([rng.uniform(-1, 1, 8000), rng.normal(.85, .002, 8000), rng.uniform(-1, 1, 8000)])
    # A small spurious plane below the real room must not become the floor.
    outliers = np.column_stack([rng.uniform(-3, 3, 80), np.full(80, -2.), rng.uniform(-3, 3, 80)])
    points = np.vstack([floor, table, outliers])
    cameras = np.array([[-1, 1.5, 0], [0, 1.5, 0], [1, 1.5, 0]])
    result = fit_floor(points, cameras, np.array([0., -1., 0.]))
    assert result.offset == pytest.approx(0., abs=.01)
    assert result.normal_down @ np.array([0., -1., 0.]) > .999
    assert result.inliers[:500].mean() > .99
    assert not result.inliers[500:].any()
