"""The static demo must keep Gaussian geometry and valid 32-byte records."""
import importlib.util
from pathlib import Path

import numpy as np

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
