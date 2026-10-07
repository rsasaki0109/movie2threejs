import numpy as np

from playworld.support import surface_patch


def test_patch_uses_only_the_support_surface_and_stays_inside_footprint():
    footprint = np.array([[0, .8, 0], [1, .8, 0], [1, .8, 1], [0, .8, 1]])
    ring = np.array([[-.05, .75, .2], [-.05, .75, .5], [-.05, .75, .8]])
    floor = ring.copy(); floor[:, 1] = 0
    background = np.vstack([ring, floor])
    rgb = np.array([[.4, .25, .1]] * 3 + [[0, 1, 0]] * 3)
    patch = surface_patch(footprint, .75, background, rgb)
    assert patch["color"] == [.4, .25, .1]
    assert patch["y"] == .752
    polygon = np.array(patch["polygon_xz"])
    assert np.all(polygon >= 0) and np.all(polygon <= 1)
    assert len(polygon) == 4


def test_no_patch_without_nearby_surface_color():
    footprint = np.array([[0, .8, 0], [1, .8, 0], [0, .8, 1]])
    assert surface_patch(footprint, .75, np.empty((0, 3)), np.empty((0, 3))) is None
    assert surface_patch(np.zeros((3, 3)), 0, np.empty((0, 3)), np.empty((0, 3))) is None


def test_patch_interpolates_observed_colors_without_using_another_height():
    footprint = np.array([[0, .8, 0], [1, .8, 0], [1, .8, 1], [0, .8, 1]])
    z = np.linspace(0, 1, 48)
    left = np.column_stack([np.full(48, -.05), np.full(48, .75), z])
    right = left.copy(); right[:, 0] = 1.05
    floor = np.vstack([left, right]); floor[:, 1] = 0
    background = np.vstack([left, right, floor])
    rgb = np.array([[.4, .2, .1]] * 48 + [[.8, .4, .2]] * 48 + [[0, 1, 0]] * 96)
    patch = surface_patch(footprint, .75, background, rgb)
    assert patch['color'] == [.6, .3, .15]
    assert patch['y'] == .752
    for position, color in zip(patch['polygon_xz'], patch['vertex_colors']):
        np.testing.assert_allclose(color, [.4, .2, .1] if position[0] == 0 else [.8, .4, .2])
