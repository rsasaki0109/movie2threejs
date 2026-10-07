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
