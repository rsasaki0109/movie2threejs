import numpy as np
import importlib.util
from pathlib import Path

from playworld.objects import View
from playworld.visual import bottom_cap, clipped_hull_fill, object_surface_mask


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


def test_clipped_visual_interior_leaves_leg_space_and_keeps_upper_outline():
    from scipy.spatial import ConvexHull
    # A sloping synthetic chair body: a wide footprint, a narrower upper back.
    points = np.array([[x, y, z] for y,width in [(0.,.3),(.9,.2)]
                       for x in [-width,width] for z in [-width,width]]) + [2,0,3]
    original = points.copy()
    centroid = np.array([2.,.45,3.])
    fill = clipped_hull_fill(points,.4,centroid)
    vertices = np.array(fill['vertices']) + centroid
    assert fill['approximate'] and fill['kind'] == 'convex_hull'
    np.testing.assert_allclose(vertices[:,1].min(),.4,atol=1e-5)
    np.testing.assert_allclose(vertices[:,1].max(),.9,atol=1e-5)
    expected_width = .3 + (.2-.3)*(.4/.9)
    at_cut = vertices[np.isclose(vertices[:,1],.4)]
    np.testing.assert_allclose(np.abs(at_cut[:,[0,2]]-[2,3]),expected_width,atol=1e-5)
    assert ConvexHull(vertices).volume < ConvexHull(points).volume
    np.testing.assert_array_equal(points,original)
    assert clipped_hull_fill(points,.9,centroid) is None
    assert clipped_hull_fill(np.zeros((10,3)),.4,centroid) is None


def test_clipped_fill_on_synthetic_room_preserves_splats_and_collision(tmp_path):
    import json
    from playworld.synthetic import make_scene
    from playworld.world import build_world
    capture,source,out = tmp_path/'capture',tmp_path/'source',tmp_path/'reviewed'
    make_scene(capture,seed=3)
    before = build_world(capture/'sparse',capture/'splats.ply',source,capture/'masks')
    saved = (source/'world.json').read_bytes()
    obj = next(item for item in before['objects'] if item['name']=='crate')
    points = np.array(obj['hull']) + obj['centroid']
    height = float((points[:,1].min()+points[:,1].max())/2)
    spec=importlib.util.spec_from_file_location('polish_visual_test',Path(__file__).parents[1]/'scripts/polish_visual_fill.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=module.polish(source,out,obj['id'],height)
    after=json.loads((out/'world.json').read_text())
    assert saved == (source/'world.json').read_bytes()
    assert before['colliders'] == after['colliders']
    assert before['support_patches'] == after['support_patches']
    for old,new in zip(before['objects'],after['objects']):
        for key in ('hull','centroid','mass','support_y'):
            assert old[key] == new[key]
        assert (source/old['splat']).read_bytes() == (out/new['splat']).read_bytes()
    assert report['collision_geometry_unchanged']


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
