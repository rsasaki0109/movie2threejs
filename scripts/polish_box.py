"""Refine one reviewed open box's rendering in an assembled PLY world, on CPU."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np

from playworld import gravity, splat_io
from playworld.colmap_io import read_model
from playworld.visual import bottom_cap, object_surface_mask
from playworld.world import install_viewer, load_masks, views_for


def polish(source: Path, sparse: Path, masks_dir: Path, out: Path, subject: int,
           background_ratio: float = .8, max_size: float = .08) -> dict:
    if out.exists():
        raise FileExistsError('Choose a fresh output directory')
    if not .5 <= background_ratio <= 1 or max_size <= 0:
        raise ValueError('Background ratio must be in [.5, 1]; maximum size must be positive')
    started = time.perf_counter()
    world = json.loads((source / 'world.json').read_text(encoding='utf-8'))
    before = copy.deepcopy(world)
    obj = next(o for o in world['objects'] if o['id'] == subject)
    if Path(obj['splat']).suffix != '.ply' or Path(world['background']).suffix != '.ply':
        raise ValueError('Input must be an assembled PLY world')
    rec = read_model(sparse)
    masks, _ = load_masks(masks_dir, [im.name for im in rec.sorted_images()])
    views = views_for(rec, masks)
    T = np.array(world['align']).reshape(4, 4).T
    props = splat_io.read_ply(source / obj['splat'])
    bg = splat_io.read_ply(source / world['background'])
    points = splat_io.means(props)
    aligned = gravity.apply(T, points)
    sizes = splat_io.max_scale(props) * np.linalg.norm(T[:3, 0])
    keep = object_surface_mask(points, aligned, sizes, views, masks, subject,
                               obj['support_y'], background_ratio, max_size,
                               visibility_points=splat_io.means(bg))
    if keep.sum() < 100:
        raise ValueError('Too few reviewed box splats remain; inspect masks/settings')
    cap = bottom_cap(aligned[keep], obj['support_y'], np.array(obj['centroid']))
    if cap is None:
        raise ValueError('Insufficient lower-side evidence for a bottom cap')
    # Discard rejected object edges; do not invent a static copy of the object.
    # Background geometry is left byte for byte intact by copytree below.
    rgb = np.column_stack([props[f'f_dc_{i}'] for i in range(3)]) * splat_io.SH_C0 + .5
    brown = (keep & (aligned[:, 1] < obj['support_y'] + .12)
             & (rgb[:, 0] > rgb[:, 1] + .02) & (rgb[:, 1] > rgb[:, 2] + .015))
    color = np.median(rgb[brown if brown.sum() >= 20 else keep], axis=0).clip(0, 1)
    obj['fill_color'] = color.round(5).tolist()
    obj['visual_fill'] = cap
    # Keep the existing observed-color support patch. Refitting it to the cap's
    # rectangle sampled the bright desk beside the dark laptop in this capture.
    # A new footprint alone is not sufficient evidence for a better hidden surface.
    removed = int((~keep).sum())
    world['stats']['gaussians'] -= removed
    world['stats']['discarded_object_gaussians'] += removed
    world['stats']['visual_cleanup'] = {'subject_id': subject, 'removed': removed,
                                      'restored_support_splats': 0}
    world['stats']['support_patches'] = len(world.get('support_patches', []))
    # Rendering refinement must not change the saved collision trajectory.
    assert world['colliders'] == before['colliders']
    for key in ['hull', 'mass', 'centroid', 'support_y']:
        assert obj[key] == next(o for o in before['objects'] if o['id'] == subject)[key]
    shutil.copytree(source, out)
    splat_io.write_ply(out / obj['splat'], splat_io.subset(props, keep))
    (out / 'world.json').write_bytes((json.dumps(world, indent=1) + '\n').encode())
    install_viewer(out)
    report = {'scope': 'CPU rendering refinement of one manually reviewed open box; no training',
              'seconds': round(time.perf_counter() - started, 3), 'subject_id': subject,
              'input_object_splats': len(points), 'kept_object_splats': int(keep.sum()),
              'removed_splats': removed, 'restored_support_splats': 0,
              'background_vote_ratio': background_ratio, 'max_splat_size_m': max_size,
              'collision_geometry_unchanged': True, 'bottom_cap': cap,
              'fill_color_srgb': obj['fill_color'], 'support_patch_replaced': False,
              'stats': world['stats']}
    (out / 'visual-refinement.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--world', type=Path, required=True)
    p.add_argument('--sparse', type=Path, required=True)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--subject', type=int, required=True)
    p.add_argument('--background-ratio', type=float, default=.8)
    p.add_argument('--max-size', type=float, default=.08)
    a = p.parse_args()
    print(json.dumps(polish(a.world, a.sparse, a.masks, a.out, a.subject,
                            a.background_ratio, a.max_size), indent=2))
