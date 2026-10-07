import numpy as np
from playworld import objects


def test_table_support_survives_stray_object_points():
    rng = np.random.default_rng(8)
    xz = rng.uniform(-.15, .15, (1200, 2))
    body = np.column_stack([xz[:, 0], rng.uniform(.76, 1, len(xz)), xz[:, 1]])
    body = np.vstack([body, [[2, -.2, 3]]])
    grid = np.arange(-.3, .301, .02)
    x, z = np.meshgrid(grid, grid)
    tabletop = np.column_stack([x.ravel(), np.full(x.size, .75), z.ravel()])
    cleaned = body[objects.inlier_mask(body)]
    assert objects.support_height(cleaned, tabletop) == .75
    assert objects.support_height(cleaned, tabletop[::200]) == 0


def test_sparse_floor_does_not_become_an_elevated_support():
    rng = np.random.default_rng(2)
    body = rng.uniform([-.2, .02, -.2], [.2, .8, .2], (1000, 3))
    floor = rng.uniform([-.4, 0, -.4], [.4, 0, .4], (3000, 3))
    assert objects.support_height(body, floor) == 0


def test_partial_box_with_noisy_bottom_stays_above_table():
    rng = np.random.default_rng(11)
    body = rng.uniform([-.2, .86, -.2], [.2, 1.2, .2], (2000, 3))
    body[:30, 1] = .79
    grid = np.arange(-.4, .401, .02)
    x, z = np.meshgrid(grid, grid)
    tabletop = np.column_stack([x.ravel(), np.full(x.size, .87), z.ravel()])
    height = objects.support_height(body, tabletop)
    assert height == .87
    obj = objects.rigid_object(body, 1, 'box', height)
    assert np.min(obj.hull[:, 1] + obj.centroid[1]) >= height + .0049
