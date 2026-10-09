"""Opt-in reviewed window photo plane; CPU only, not exterior reconstruction.

Requires undistorted calibrated photographs and manually reviewed plane/bounds.
Copies the world into a fresh output, keeps physics and movable objects intact,
and replaces only the configured background strip. No image generation occurs.
The strip uses world-x offsets relative to x = slope*z + intercept, not normal
distance. All bounds depend on the existing world's assumed scale.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from playworld import gravity, splat_io
from playworld.colmap_io import read_model
from playworld.world import install_viewer


def window_grid(slope, intercept, z_range, y_range, resolution):
    width, height = resolution
    settings = [slope, intercept, *z_range, *y_range]
    if (not np.isfinite(settings).all() or z_range[0] >= z_range[1]
            or y_range[0] >= y_range[1] or width < 2 or height < 2):
        raise ValueError('invalid window plane, bounds or texture resolution')
    z, y = np.meshgrid(np.linspace(*z_range, width), np.linspace(y_range[1], y_range[0], height))
    return np.stack([slope*z+intercept, y, z], -1)


def sample_photo(world_points, transform, image, camera, pixels):
    if camera.model != 'PINHOLE':
        raise ValueError('backdrop requires undistorted PINHOLE photographs')
    raw = gravity.apply(np.linalg.inv(transform), world_points)
    cp = raw @ image.R.T + image.tvec
    height, width = pixels.shape[:2]
    K = camera.K.copy()
    K[0] *= width/camera.width
    K[1] *= height/camera.height
    projected = cp @ K.T
    uv = projected[:, :2]/np.maximum(projected[:, 2:], 1e-12)
    u, v = uv.T
    if not ((cp[:, 2] > 0) & (u >= 0) & (v >= 0) & (u <= width-1) & (v <= height-1)).all():
        raise ValueError('reviewed photo does not cover its entire window patch')
    return np.stack([map_coordinates(pixels[:, :, c].astype(float), [v, u],
                                    order=1, mode='nearest') for c in range(3)], axis=1)


def bake(scene, world, config, out):
    if out.exists():
        raise FileExistsError('Choose a fresh output directory')
    started = time.perf_counter()
    settings = json.loads(config.read_text(encoding='utf-8'))
    source = json.loads((world/'world.json').read_text(encoding='utf-8'))
    T = np.asarray(source['align']).reshape(4, 4, order='F')
    linear = T[:3, :3]
    scale = np.linalg.norm(linear[:, 0])
    if (not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]) or scale <= 0
            or not np.allclose(linear.T@linear, np.eye(3)*scale**2, rtol=1e-5, atol=1e-8)):
        raise ValueError('world alignment must be a finite similarity transform')
    patches = settings['patches']
    for left, right in zip(patches, patches[1:]):
        if left['z'][1] != right['z'][0]:
            raise ValueError('window patches must be contiguous and ordered')
    slope, intercept = settings['plane']['slope'], settings['plane']['intercept']
    z_range = [patches[0]['z'][0], patches[-1]['z'][1]]
    y_range = settings['y']
    points = window_grid(slope, intercept, z_range, y_range, settings['resolution'])
    flat = points.reshape(-1, 3)
    atlas = np.zeros((len(flat), 3))
    covered = np.zeros(len(flat), dtype=bool)
    rec = read_model(scene/'sparse')
    images = {im.name:im for im in rec.sorted_images()}
    def file_info(path):
        return {'bytes':path.stat().st_size, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    photos = {}
    for index, patch in enumerate(patches):
        low, high = patch['z']
        if low >= high:
            raise ValueError('window patches must have positive width')
        selected = (flat[:, 2] >= low) & (flat[:, 2] < high if index < len(patches)-1 else flat[:, 2] <= high)
        image = images[patch['image']]
        path = (scene/'images'/image.name).resolve()
        if not path.is_relative_to((scene/'images').resolve()):
            raise ValueError('image path escaped scene')
        pixels = np.asarray(Image.open(path).convert('RGB'))
        atlas[selected] = sample_photo(flat[selected], T, image, rec.cameras[image.camera_id], pixels)
        covered[selected] = True
        photos[image.name] = file_info(path)
    if not covered.all():
        raise ValueError('window patch coverage is incomplete')
    relative = Path(source['background'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('unsafe background path')
    props = splat_io.read_ply(world/relative)
    xyz = gravity.apply(T, splat_io.means(props))
    signed = xyz[:, 0] - (slope*xyz[:, 2]+intercept)
    strip = settings['interior_strip_m']
    padding = settings.get('cut_height_padding_m', 0)
    if not np.isfinite([*strip, padding]).all() or strip[0] >= strip[1] or padding < 0:
        raise ValueError('invalid replacement strip or height padding')
    cut = ((signed > strip[0]) & (signed < strip[1]) & (xyz[:, 1] > y_range[0])
           & (xyz[:, 1] < y_range[1]+padding) & (xyz[:, 2] > z_range[0]) & (xyz[:, 2] < z_range[1]))
    if cut.all():
        raise ValueError('window replacement would leave an empty background')
    shutil.copytree(world, out)
    install_viewer(out)  # Refresh older captured worlds so they can render photo planes.
    target = out/'window-photo.jpg'
    Image.fromarray(np.clip(atlas.reshape(*points.shape[:2], 3), 0, 255).astype('uint8')).save(target, quality=95)
    splat_io.write_ply(out/relative, splat_io.subset(props, ~cut))
    zl, zh = z_range
    yl, yh = y_range
    source.setdefault('photo_backdrops', []).append({'texture':target.name,
        'vertices':[[slope*zl+intercept, yl, zl], [slope*zl+intercept, yh, zl],
                    [slope*zh+intercept, yh, zh], [slope*zh+intercept, yl, zh]],
        'uv':[[0, 0], [0, 1], [1, 1], [1, 0]], 'label':settings['label'],
        'source_photographs':list(photos)})
    source['stats']['gaussians'] -= int(cut.sum())
    source['stats']['window_photo_replaced_gaussians'] = int(cut.sum())
    (out/'world.json').write_text(json.dumps(source, indent=2)+'\n', encoding='utf-8')
    report = {'format':'playworld-reviewed-window-photo/1', 'gpu_used':False,
              'exterior_reconstructed_3d':False, 'generated_imagery':False,
              'settings':settings, 'source_world':file_info(world/'world.json'),
              'source_config':file_info(config), 'source_photographs':photos,
              'removed_background_gaussians':int(cut.sum()), 'coverage':float(covered.mean()),
              'physics_unchanged':True, 'movable_objects_unchanged':True,
              'texture':file_info(target), 'background':file_info(out/relative),
              'seconds':time.perf_counter()-started}
    (out/'window-photo-provenance.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['scene', 'world', 'config', 'out']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(bake(a.scene, a.world, a.config, a.out), indent=2))
