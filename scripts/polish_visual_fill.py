"""Clip a reviewed object's approximate visual fill; preserve its physics body."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from playworld.visual import clipped_hull_fill


def polish(source: Path, out: Path, subject: int, min_y: float):
    source, out = source.resolve(), out.resolve()
    if out.is_relative_to(source):
        raise ValueError('Choose a fresh output directory outside the source world')
    world = json.loads((source / 'world.json').read_text())
    obj = next(item for item in world['objects'] if item['id'] == subject)
    fill = clipped_hull_fill(np.array(obj['hull']) + obj['centroid'], min_y, np.array(obj['centroid']))
    if fill is None:
        raise ValueError('Cut height leaves no solid visual interior')
    if not obj.get('fill_color'):
        raise ValueError('Object has no reviewed interior color')
    shutil.copytree(source, out)
    obj['visual_fill'] = fill
    (out / 'world.json').write_text(json.dumps(world, indent=2)+'\n', encoding='utf-8')
    report = {'subject_id': subject, 'min_world_y': min_y, 'approximate': True,
              'collision_geometry_unchanged': True, 'source_world_sha256': hashlib.sha256((source/'world.json').read_bytes()).hexdigest()}
    (out / 'visual-fill-review.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--object', type=int, required=True)
    parser.add_argument('--min-y', type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(polish(args.world, args.out, args.object, args.min_y), indent=2))
