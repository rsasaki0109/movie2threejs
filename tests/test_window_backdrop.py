"""Reviewed photo-plane projection and bounded background replacement."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image as PillowImage
import pytest
from scipy.spatial.transform import Rotation

from playworld import gravity, splat_io
from playworld.colmap_io import Camera, Image, Reconstruction, write_model

spec = importlib.util.spec_from_file_location('bake_window', Path(__file__).parents[1]/'scripts/bake_window_backdrop.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def synthetic_window():
    T = np.eye(4)
    T[:3, :3] = 1.7*Rotation.from_euler('y', .3).as_matrix()
    T[:3, 3] = [.2, .3, .4]
    R = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    raw_R = R@T[:3, :3]/1.7
    image = Image(1, Rotation.from_matrix(raw_R).as_quat()[[3, 0, 1, 2]], R@T[:3, 3]/1.7, 1, 'window.png')
    camera = Camera(1, 'PINHOLE', 200, 200, np.array([80, 80, 100, 100]))
    y, x = np.indices((100, 100))
    pixels = np.stack([x, y, np.full_like(x, 100)], -1).astype('uint8')
    return T, image, camera, pixels


def test_window_projection_keeps_rays_under_world_alignment_and_image_resize():
    T, image, camera, pixels = synthetic_window()
    grid = module.window_grid(0, 2, [-.5, .5], [-.5, .5], [21, 21])
    colors = module.sample_photo(grid.reshape(-1, 3), T, image, camera, pixels).reshape(21, 21, 3)
    np.testing.assert_allclose(colors[0, 0], [40, 40, 100], atol=1e-6)
    np.testing.assert_allclose(colors[-1, -1], [60, 60, 100], atol=1e-6)
    np.testing.assert_allclose(colors[10, 10], [50, 50, 100], atol=1e-6)
    with pytest.raises(ValueError, match='entire window patch'):
        module.sample_photo(np.array([[2, 0, 10]]), T, image, camera, pixels)
    with pytest.raises(ValueError, match='entire window patch'):
        module.sample_photo(np.array([[-2, 0, 0]]), T, image, camera, pixels)


def test_backdrop_replaces_only_reviewed_strip_and_preserves_room_physics_and_objects(tmp_path):
    T, image, camera, pixels = synthetic_window()
    scene, world = tmp_path/'scene', tmp_path/'world'
    (scene/'images').mkdir(parents=True)
    PillowImage.fromarray(pixels).save(scene/'images/window.png')
    write_model(Reconstruction({1:camera}, {1:image}, np.array([[0., 0, 1]]),
                               np.zeros((1, 3), dtype='uint8')), scene/'sparse')
    world.mkdir()
    # Inside reviewed strip; below it; outside z extent; outside normal extent.
    points = np.array([[2.2, 0, 0], [2.2, -1, 0], [2.2, 0, 1], [3.2, 0, 0]])
    props = splat_io.make_splats(gravity.apply(np.linalg.inv(T), points), np.full((4, 3), .5), .01)
    splat_io.write_ply(world/'background.ply', props)
    (world/'chair.ply').write_bytes(b'reviewed movable chair unchanged')
    source = {'align':T.T.reshape(-1).tolist(), 'background':'background.ply',
              'colliders':[{'center':[2, 0, 0], 'half_extents':[.1, 1, 1]}],
              'objects':[{'splat':'chair.ply'}], 'stats':{'gaussians':4}}
    (world/'world.json').write_text(json.dumps(source), encoding='utf-8')
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'plane':{'slope':0, 'intercept':2}, 'y':[-.5, .5],
        'patches':[{'z':[-.5, .5], 'image':'window.png'}], 'resolution':[21, 21],
        'interior_strip_m':[-.05, .5], 'label':'Source photo plane'}), encoding='utf-8')
    out = tmp_path/'baked'
    report = module.bake(scene, world, config, out)
    assert report['removed_background_gaussians'] == 1
    assert report['coverage'] == 1 and not report['exterior_reconstructed_3d']
    result = json.loads((out/'world.json').read_text())
    assert result['objects'] == source['objects'] and result['colliders'] == source['colliders']
    assert (out/'chair.ply').read_bytes() == (world/'chair.ply').read_bytes()
    np.testing.assert_allclose(splat_io.means(splat_io.read_ply(out/'background.ply')),
                               splat_io.means(splat_io.read_ply(world/'background.ply'))[1:])
    assert result['photo_backdrops'][0]['source_photographs'] == ['window.png']
    assert result['stats']['gaussians'] == 3
    with pytest.raises(FileExistsError):
        module.bake(scene, world, config, out)
