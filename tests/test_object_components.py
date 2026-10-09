import numpy as np
import pytest

from playworld import objects
from playworld.synthetic import make_scene
from playworld.world import build_world


def chair_points():
    """A dense backrest, much sparser attached seat, post and feet."""
    rng = np.random.default_rng(19)
    back = rng.uniform([0, .54, -.25], [.015, 1.2, .25], (6000, 3))
    x, z = np.meshgrid(np.arange(0, .501, .04), np.arange(-.24, .241, .04))
    seat = np.column_stack([x.ravel(), np.full(x.size, .55), z.ravel()])
    post = np.column_stack([np.full(14, .25), np.linspace(.03, .55, 14), np.zeros(14)])
    feet = np.array([[.25 + d * np.cos(a), .03, d * np.sin(a)]
                     for a in np.linspace(0, 2*np.pi, 5, endpoint=False)
                     for d in np.arange(0, .301, .04)])
    return np.vstack([back, seat, post, feet])


def test_sparse_chair_parts_survive_while_detached_spill_is_removed():
    chair = chair_points()
    rng = np.random.default_rng(4)
    spill = rng.uniform([3, 0, 3], [3.5, .5, 3.5], (500, 3))
    points = np.vstack([chair, spill])
    keep = objects.connected_inlier_mask(points)
    assert keep[:len(chair)].all()
    assert not keep[len(chair):].any()
    # Re-filtering an already cleaned chair must not erode its sparse parts.
    assert objects.connected_inlier_mask(points[keep]).all()
    permutation = rng.permutation(len(points))
    np.testing.assert_array_equal(objects.connected_inlier_mask(points[permutation]), keep[permutation])
    body = objects.rigid_object(points, 1, 'chair', method='connected')
    assert body is not None
    np.testing.assert_allclose(np.ptp(body.hull, axis=0),
                               np.ptp(np.vstack([chair, [chair[0, 0], .005, chair[0, 2]]]), axis=0))


def test_broad_splats_cannot_bridge_separate_instances():
    chair = chair_points()
    bridge = np.column_stack([np.arange(.6, 1.101, .06), np.full(9, .55), np.zeros(9)])
    other = np.array([[1.16, .55, 0], [1.18, .55, 0]])
    points = np.vstack([chair, bridge, other])
    sizes = np.r_[np.full(len(chair), .01), np.full(len(bridge), .4), [.01, .01]]
    keep = objects.movable_splat_mask(points, sizes, method='connected')
    assert keep[:len(chair)].all()
    assert not keep[len(chair):].any()


def test_component_filter_handles_missing_and_nonfinite_points():
    assert objects.connected_inlier_mask(np.empty((0, 3))).shape == (0,)
    points = np.array([[0, 0, 0], [.01, 0, 0], [np.nan, 0, 0], [0, np.inf, 0]])
    assert objects.connected_inlier_mask(points).tolist() == [True, True, False, False]
    assert not objects.connected_inlier_mask(points[2:]).any()


@pytest.mark.parametrize('cell', [0, -.1, np.nan, np.inf])
def test_component_filter_rejects_invalid_scale(cell):
    with pytest.raises(ValueError, match='cell'):
        objects.connected_inlier_mask(np.zeros((8, 3)), cell=cell)


def test_connected_filter_preserves_synthetic_room_objects_and_support(tmp_path):
    make_scene(tmp_path/'capture', seed=3)
    world = build_world(tmp_path/'capture/sparse', tmp_path/'capture/splats.ply',
                        tmp_path/'out', tmp_path/'capture/masks', object_filter='connected')
    by = {o['name']: o for o in world['objects']}
    assert set(by) == {'chair', 'crate', 'mug'}
    assert world['stats']['object_filter'] == 'connected'
    assert by['mug']['support_y'] == pytest.approx(.75 * 1.5 / 1.55, abs=.04)
    assert by['chair']['support_y'] == by['crate']['support_y'] == 0
    assert np.ptp(by['chair']['hull'], axis=0)[1] > .8
    for obj in by.values():
        h = np.array(obj['hull']) + obj['centroid']
        assert h[:, 1].min() >= obj['support_y'] + .0048
        for box in world['colliders'][1:]:
            c, e = np.array(box['center']), np.array(box['half_extents'])
            overlap = np.minimum(c+e, h.max(axis=0)) - np.maximum(c-e, h.min(axis=0))
            assert np.any(overlap <= 1e-6), obj['name']
