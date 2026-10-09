import numpy as np
import pytest

from playworld.cleanup import surface_splat_mask


def test_cleanup_preserves_thin_walls_and_translucent_detail():
    scales = np.array([[.4, .4, .002], [.02, .002, .01], [.01, .01, .01], [.12, .1, .03], [.09, .09, .09]])
    opacity = np.array([.9, .02, .95, .3, .99])
    np.testing.assert_array_equal(surface_splat_mask(scales, opacity), [True, True, True, False, False])
    # The covariance axes may be permuted by rotation; the decision is invariant.
    np.testing.assert_array_equal(surface_splat_mask(scales[:, ::-1], opacity), [True, True, True, False, False])


def test_cleanup_rejects_nonfinite_and_nonphysical_covariance():
    scales = np.array([[.01, .01, .01], [np.nan, .01, .01], [.01, -.01, .01], [.01, .01, .01]])
    np.testing.assert_array_equal(surface_splat_mask(scales, [.5, .5, .5, np.inf]), [True, False, False, False])
    assert surface_splat_mask(np.empty((0, 3)), np.empty(0)).shape == (0,)


def test_cleanup_checks_correspondence():
    with pytest.raises(ValueError):
        surface_splat_mask(np.ones((2, 3)), np.ones(1))


def test_world_cleanup_keeps_collision_geometry_and_reports_removal(tmp_path):
    from playworld import splat_io
    from playworld.synthetic import make_scene
    from playworld.world import build_world
    root = tmp_path / "capture"
    truth = make_scene(root, seed=3)
    props = splat_io.read_ply(root / "splats.ply")
    point = truth["scale"] * (np.array([[0, 2, -1]]) @ truth["R"].T) + truth["t"]
    noise = splat_io.make_splats(point, np.array([[.5, .5, .5]]), .15 * truth["scale"], alpha=.4)
    augmented = {key: np.r_[values, noise[key]] for key, values in props.items()}
    splat_io.write_ply(root / "noisy.ply", augmented)
    original = build_world(root / "sparse", root / "noisy.ply", tmp_path / "original", copy_viewer=False)
    cleaned = build_world(root / "sparse", root / "noisy.ply", tmp_path / "clean", copy_viewer=False, clean_splats=True)
    assert cleaned["stats"]["discarded_diffuse_gaussians"] == 1
    assert cleaned["stats"]["gaussians"] == original["stats"]["gaussians"] - 1
    assert cleaned["colliders"] == original["colliders"]


def test_interior_cleanup_preserves_observed_room_surfaces_and_outdoor_geometry():
    from playworld.cleanup import interior_support_mask
    cameras = np.array([[-2, 1.5, -2], [2, 1.5, -2], [2, 1.5, 2], [-2, 1.5, 2]])
    # Synthetic floor, ceiling, observed chair, unsupported interior floater,
    # unsupported exterior scenery, and a point on the footprint boundary.
    reference = np.array([[0, 0, 0], [.3, 1, .3], [1, 0, 1]])
    points = np.array([[0, 0, 0], [0, 3, 0], [.3, 1, .3],
                       [0, 1.6, 0], [5, 1.6, 0], [-1.98, 1.6, 0]])
    original = points.copy()
    keep = interior_support_mask(points, reference, cameras)
    np.testing.assert_array_equal(keep, [True, True, True, False, True, True])
    np.testing.assert_array_equal(points, original)
    # A supported point within the threshold must survive, including a thin part.
    assert interior_support_mask(np.array([[.3, 1.29, .3]]), reference, cameras)[0]
    assert interior_support_mask(points[keep], reference, cameras).all()


def test_interior_cleanup_requires_a_valid_footprint_and_finite_reference():
    from playworld.cleanup import interior_support_mask
    points = np.array([[0, 1, 0]])
    cameras = np.array([[-1, 1.5, -1], [1, 1.5, -1], [0, 1.5, 1]])
    with pytest.raises(ValueError, match='nonzero horizontal area'):
        interior_support_mask(points, points, np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2]]))
    with pytest.raises(ValueError, match='reference points'):
        interior_support_mask(points, np.empty((0, 3)), cameras)
    with pytest.raises(ValueError, match='finite Nx3'):
        interior_support_mask(points, np.array([[np.nan, 0, 0]]), cameras)
    with pytest.raises(ValueError, match='invalid interior support'):
        interior_support_mask(points, points, cameras, margin=-.1)
    assert interior_support_mask(np.empty((0, 3)), points, cameras).shape == (0,)


def test_interior_tool_preserves_a_complete_synthetic_room_and_original_inputs(tmp_path):
    import hashlib
    import importlib.util
    import json
    from pathlib import Path
    from playworld import splat_io
    from playworld.synthetic import make_scene

    scene = tmp_path/'scene'
    truth = make_scene(scene, seed=3)
    props = splat_io.read_ply(scene/'splats.ply')
    # One unsupported room-center floater and exterior geometry. The real
    # synthetic room, its floor and all observed rigid-body parts must survive.
    fake_world = np.array([[0, 1.5, 0], [12, 1.5, 0]])
    fake_capture = truth['scale']*fake_world@truth['R'].T + truth['t']
    fake = splat_io.make_splats(fake_capture, np.full((2, 3), .5), .01*truth['scale'])
    augmented = {k: np.r_[v, fake[k]] for k, v in props.items()}
    input_ply = scene/'augmented.ply'
    splat_io.write_ply(input_ply, augmented)
    before = hashlib.sha256(input_ply.read_bytes()).hexdigest()
    T = np.eye(4)
    T[:3, :3] = truth['R'].T/truth['scale']
    T[:3, 3] = -T[:3, :3]@truth['t']
    world = tmp_path/'world.json'
    world.write_text(json.dumps({'align': T.T.reshape(-1).tolist()}), encoding='utf-8')
    spec = importlib.util.spec_from_file_location('clean_interior', Path(__file__).parents[1]/'scripts/clean_interior.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path/'cleaned'
    report = module.clean(scene, input_ply, world, out)
    assert report['removed_gaussians'] == 1
    np.testing.assert_array_equal(np.load(out/'keep.npy'), np.r_[np.ones(len(props['x']), dtype=bool), False, True])
    assert hashlib.sha256(input_ply.read_bytes()).hexdigest() == before
    result = splat_io.read_ply(out/'cleaned.ply')
    serialized_input = splat_io.read_ply(input_ply)
    for k in props:
        np.testing.assert_array_equal(result[k], serialized_input[k][np.load(out/'keep.npy')])
    with pytest.raises(FileExistsError):
        module.clean(scene, input_ply, world, out)
