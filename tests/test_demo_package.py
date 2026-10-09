"""The static demo must keep Gaussian geometry and valid 32-byte records."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from playworld import splat_io


def test_splat_package_roundtrip(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/build_demo.py"
    spec = importlib.util.spec_from_file_location("build_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    points = np.array([[1, -2, 3], [-.25, .5, 1.75]])
    colors = np.array([[.1, .5, .9], [1, 0, .25]])
    props = splat_io.make_splats(points, colors, scale=.025, alpha=.8)
    # Check a non-identity wxyz rotation, not just all-zero imaginary parts.
    props["rot_0"] = np.array([1, 2**-.5])
    props["rot_2"] = np.array([0, 2**-.5])
    source, packed = tmp_path / "source.ply", tmp_path / "packed.splat"
    splat_io.write_ply(source, props)
    assert module.pack_splat(source, packed) == 2
    rows = np.frombuffer(packed.read_bytes(), dtype=np.uint8).reshape(2, 32)
    np.testing.assert_allclose(rows[:, :12].copy().view("<f4"), points)
    np.testing.assert_allclose(rows[:, 12:24].copy().view("<f4"), .025)
    np.testing.assert_allclose(rows[:, 24:27] / 255, colors, atol=1/255)
    np.testing.assert_allclose(rows[:, 27] / 255, .8, atol=1/255)
    rotations = (rows[:, 28:32].astype(float) - 128) / 128
    np.testing.assert_allclose(rotations, [[1, 0, 0, 0], [2**-.5, 0, 2**-.5, 0]], atol=1/128)


def test_spz_preserves_capture_axes_and_spherical_harmonics(tmp_path):
    spz = pytest.importorskip("spz")
    path = Path(__file__).resolve().parents[1] / "scripts/build_demo.py"
    spec = importlib.util.spec_from_file_location("build_demo_spz", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    points = np.array([[1, -2, 3], [-.25, .5, 1.75]])
    props = splat_io.make_splats(points, np.array([[.2, .5, .8], [.4, .6, .3]]), .025, alpha=.8)
    props['rot_0'] = np.array([1., 2**-.5])
    props['rot_2'] = np.array([0., 2**-.5])
    for index in range(45):
        props[f"f_rest_{index}"] = np.full(2, .2 if index == 0 else 0)
    source, packed = tmp_path / "source.ply", tmp_path / "packed.spz"
    splat_io.write_ply(source, props)
    assert module.pack_spz(source, packed) == 2
    decoded = spz.load_spz(str(packed))
    assert decoded.sh_degree == 3
    np.testing.assert_allclose(decoded.positions.reshape(-1, 3), points, atol=1/4096)
    np.testing.assert_allclose(np.exp(decoded.scales.reshape(-1, 3)), .025, rtol=.04)
    actual = decoded.rotations.reshape(-1, 4)[:, [3, 0, 1, 2]]
    expected_rotation = np.array([[1., 0, 0, 0], [2**-.5, 0, 2**-.5, 0]])
    # Quaternion signs may differ; both represent the same orientation.
    np.testing.assert_allclose(np.abs((actual * expected_rotation).sum(axis=1)), 1, atol=.001)
    # PLY stores channels in blocks; SPZ stores channels inside coefficients.
    expected = np.zeros((2, 15, 3)); expected[:, 0, 0] = .2
    np.testing.assert_allclose(decoded.sh.reshape(2, 15, 3), expected, atol=.065)


def test_demo_packages_reviewed_photo_texture_without_changing_its_bytes(tmp_path):
    import json
    spec = importlib.util.spec_from_file_location('build_demo_photo', Path(__file__).parents[1]/'scripts/build_demo.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, out = tmp_path/'source', tmp_path/'packed'
    source.mkdir()
    splat_io.write_ply(source/'background.ply', splat_io.make_splats(np.array([[0., 0, 1]]), np.full((1, 3), .5), .01))
    (source/'window.jpg').write_bytes(b'exact reviewed photo bytes')
    world = {'background':'background.ply', 'objects':[], 'colliders':[],
             'photo_backdrops':[{'texture':'window.jpg', 'label':'Source photo plane'}]}
    (source/'world.json').write_text(json.dumps(world), encoding='utf-8')
    module.package_world(source, out)
    assert (out/'window.jpg').read_bytes() == (source/'window.jpg').read_bytes()
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['photo_backdrops'][0]['path'] == 'window.jpg'
    assert manifest['gaussians'] == 1
