"""Opt-in sparse-support cleanup inside the captured camera footprint; CPU only.

Uses an existing world's y-up alignment/assumed scale. This heuristic can remove
untracked interior surfaces. Original inputs remain intact in a fresh output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from playworld import gravity, splat_io
from playworld.cleanup import floor_footprint, interior_support_mask
from playworld.colmap_io import read_model


def clean(scene, splats, world, out, distance=.3, margin=.05,
          min_height=.25, max_height=2.75, test_every=0,
          footprint='cameras', floor_band=.06, floor_cell=.2):
    if out.exists():
        raise FileExistsError('Choose a fresh output directory')
    if test_every != 0 and test_every < 2:
        raise ValueError('test-every must be zero or at least two')
    if footprint not in ('cameras', 'floor'):
        raise ValueError('footprint must be cameras or floor')
    started = time.perf_counter()
    rec = read_model(scene / 'sparse')
    images = rec.sorted_images()
    chosen = [im for i, im in enumerate(images) if not test_every or i % test_every]
    transform = np.asarray(json.loads(world.read_text(encoding='utf-8'))['align']).reshape(4, 4, order='F')
    if (not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1])):
        raise ValueError('world alignment must be a finite affine matrix')
    linear = transform[:3, :3]
    scale = np.linalg.norm(linear[:, 0])
    if scale <= 0 or not np.allclose(linear.T @ linear, np.eye(3)*scale**2, rtol=1e-5, atol=1e-8):
        raise ValueError('world alignment must be a similarity transform')
    props = splat_io.read_ply(splats)
    aligned = gravity.apply(transform, splat_io.means(props))
    reference = gravity.apply(transform, rec.xyz)
    centers = gravity.apply(transform, np.array([im.center for im in chosen]).reshape(-1, 3))
    anchors = floor_footprint(reference, floor_band, floor_cell) if footprint == 'floor' else centers
    keep = interior_support_mask(aligned, reference, anchors, distance, margin, min_height, max_height)
    if not keep.any():
        raise ValueError('cleanup would leave an empty Gaussian cloud')
    out.mkdir(parents=True)
    target = out / 'cleaned.ply'
    splat_io.write_ply(target, splat_io.subset(props, keep))
    np.save(out / 'keep.npy', keep)
    def file_info(path):
        return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    report = {'format': 'playworld-interior-support-cleanup/1', 'gpu_used': False,
              'input_gaussians': len(keep), 'retained_gaussians': int(keep.sum()),
              'removed_gaussians': int((~keep).sum()), 'reference_points': len(reference),
              'reference_cameras': [im.name for im in chosen] if footprint == 'cameras' else [],
              'footprint_uses_camera_centers': footprint == 'cameras',
              'settings': {'max_reference_distance_m': distance, 'footprint_inset_m': margin,
                           'min_height_m': min_height, 'max_height_m': max_height,
                           'test_every': test_every, 'footprint': footprint,
                           'floor_band_m': floor_band if footprint == 'floor' else None,
                           'floor_voxel_m': floor_cell if footprint == 'floor' else None},
              'footprint_anchor_points': len(anchors),
              'alignment': transform.T.reshape(-1).tolist(), 'scale_assumed': True,
              'limitation': 'Footprint is not measured free space; unsupported interior surfaces can be removed. Floor mode can extend beyond camera centers.',
              'inputs': {'splats': file_info(splats), 'world': file_info(world),
                         'sparse': {p.name: file_info(p) for p in sorted((scene/'sparse').glob('*.bin'))}},
              'outputs': {'cleaned.ply': file_info(target), 'keep.npy': file_info(out/'keep.npy')},
              'seconds': time.perf_counter() - started}
    (out / 'cleanup.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True, help='Full camera/point scene')
    p.add_argument('--splats', type=Path, required=True)
    p.add_argument('--world', type=Path, required=True, help='Existing world.json for alignment/assumed scale')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--distance', type=float, default=.3)
    p.add_argument('--margin', type=float, default=.05)
    p.add_argument('--min-height', type=float, default=.25)
    p.add_argument('--max-height', type=float, default=2.75)
    p.add_argument('--test-every', type=int, default=0, help='Exclude every Nth camera to match a validation split; zero uses all')
    p.add_argument('--footprint', choices=['cameras', 'floor'], default='cameras',
                   help='Opt-in observed-floor footprint; default uses camera centers')
    p.add_argument('--floor-band', type=float, default=.06, help='Floor point band around aligned y=0, assumed meters')
    p.add_argument('--floor-cell', type=float, default=.2, help='Horizontal floor connectivity voxel, assumed meters')
    a = p.parse_args()
    print(json.dumps(clean(a.scene, a.splats, a.world, a.out, a.distance, a.margin,
                           a.min_height, a.max_height, a.test_every,
                           a.footprint, a.floor_band, a.floor_cell), indent=2))
