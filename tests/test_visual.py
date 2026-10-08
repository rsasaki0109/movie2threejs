import numpy as np
import importlib.util
from pathlib import Path

from playworld.objects import View
from playworld.visual import bottom_cap, object_surface_mask


def test_render_cleanup_keeps_contact_but_rejects_table_and_mask_spill():
    # View looks down +z; box, tabletop, outside-mask and broad points occupy
    # separate pixels, so rejection cannot be explained by z-buffer occlusion.
    points = np.array([[0, .1, 1], [.1, -.01, 1], [-.1, .1, 1], [.1, .1, 1]])
    K = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
    view = View(K, np.eye(3), np.zeros(3), 100, 100)
    mask = np.zeros((100, 100), int)
    mask[45:65, 45:65] = 1
    world = points.copy(); world[:, 1] += .75
    keep = object_surface_mask(points, world, np.array([.02, .02, .02, .12]),
                               [view, view], [mask, mask], 1, .75)
    assert keep.tolist() == [True, False, False, False]


def test_box_bottom_is_horizontal_rotated_inset_and_not_a_lid():
    rng = np.random.default_rng(8)
    angle = .37
    c, s = np.cos(angle), np.sin(angle)
    axes = np.array([[c, -s], [s, c]])
    # Dense lower surfaces of a synthetic box, with protruding upper contents.
    xz = rng.uniform([-.2, -.3], [.2, .3], (3000, 2)) @ axes.T + [2, 3]
    points = np.column_stack([xz[:, 0], rng.uniform(.76, .85, 3000), xz[:, 1]])
    contents = np.array([[1.7, 1.4, 3.5], [2.7, 1.7, 3.9]])
    centroid = np.array([2., .9, 3.])
    cap = bottom_cap(np.vstack([points, contents]), .75, centroid)
    vertices = np.array(cap['vertices']) + centroid
    np.testing.assert_allclose(vertices[:, 1], .757)
    local = (vertices[:, [0, 2]] - [2, 3]) @ axes
    assert np.all(np.abs(local) < [.2, .3])
    assert np.ptp(local, axis=0).min() > .35
    assert cap['kind'] == 'bottom_cap' and cap['approximate']


def test_occluded_surfaces_are_not_removed_for_missing_mask_votes():
    points = np.array([[0., 0, 2]])
    K = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
    view = View(K, np.eye(3), np.zeros(3), 100, 100)
    mask = np.zeros((100, 100), int)
    front = np.array([[0., 0, 1]])
    keep = object_surface_mask(points, points, np.array([.02]), [view, view],
                               [mask, mask], 1, 0, visibility_points=front)
    assert keep.tolist() == [True]
    assert not object_surface_mask(points, points, np.array([.02]), [view, view],
                                   [mask, mask], 1, 0).any()


def test_no_bottom_when_support_evidence_is_absent_or_degenerate():
    assert bottom_cap(np.zeros((0, 3)), .75, np.zeros(3)) is None
    assert bottom_cap(np.zeros((30, 3)), 0, np.zeros(3)) is None


def test_polishing_a_synthetic_box_preserves_collision_bodies_and_source(tmp_path):
    from playworld.synthetic import make_scene
    from playworld.world import build_world
    capture, source = tmp_path / 'capture', tmp_path / 'source'
    make_scene(capture, seed=3)
    before = build_world(capture / 'sparse', capture / 'splats.ply', source, capture / 'masks')
    saved = (source / 'world.json').read_bytes()
    subject = next(o['id'] for o in before['objects'] if o['name'] == 'crate')
    path = Path(__file__).resolve().parents[1] / 'scripts/polish_box.py'
    spec = importlib.util.spec_from_file_location('polish_box_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    report = module.polish(source, capture / 'sparse', capture / 'masks', tmp_path / 'out', subject)
    import json
    after = json.loads((tmp_path / 'out/world.json').read_text())
    assert (source / 'world.json').read_bytes() == saved
    assert after['colliders'] == before['colliders']
    assert after['support_patches'] == before['support_patches']
    assert (tmp_path / 'out' / before['background']).read_bytes() == (source / before['background']).read_bytes()
    for old, new in zip(before['objects'], after['objects']):
        for key in ['id', 'centroid', 'hull', 'mass', 'support_y']:
            assert new[key] == old[key]
    assert report['collision_geometry_unchanged']
    assert next(o for o in after['objects'] if o['id'] == subject)['visual_fill']['kind'] == 'bottom_cap'
