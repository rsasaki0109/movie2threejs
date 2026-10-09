"""Apply reviewed cross-frame instance associations without rerunning SAM 3.

Writes a new assembly-only scene (sparse/, masks/, instance-review.json).
It does not copy photographs, change the source masks or infer associations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from playworld.colmap_io import Reconstruction, read_model, write_model


def remap_instances(scene: Path, review_path: Path, out: Path) -> dict:
    scene, out = Path(scene).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    labels, frames = review['labels'], review['frames']
    if not labels or not frames or any(not str(k).isdigit() or not 1 <= int(k) <= np.iinfo(np.int32).max
                                      or str(int(k)) != str(k) for k in labels):
        raise ValueError('review requires positive object labels and selected frames')
    if any(not isinstance(name, str) or not name for name in labels.values()):
        raise ValueError('object names must be nonempty strings')
    rec = read_model(scene/'sparse')
    images = {im.name: im for im in rec.images.values()}
    if not set(frames).issubset(images):
        raise ValueError('review names a frame outside the camera model')
    masks, receipts = {}, {}
    for name, mapping in frames.items():
        path = (scene/'masks'/Path(name).with_suffix('.npy')).resolve()
        if not path.is_relative_to(scene/'masks'):
            raise ValueError('mask path escaped the source scene')
        source = np.load(path, allow_pickle=False)
        cam = rec.cameras[images[name].camera_id]
        if source.ndim != 2 or not np.issubdtype(source.dtype, np.integer) or (source < 0).any():
            raise ValueError('source masks must be nonnegative 2D integer arrays')
        # Downsampled masks are valid: world.views_for scales the intrinsics.
        if not source.size or min(cam.width, cam.height) <= 0:
            raise ValueError('empty mask or camera')
        target = np.zeros(source.shape, dtype=np.int32)
        if not mapping:
            raise ValueError('selected frames need at least one association')
        for original, canonical in mapping.items():
            if not str(original).isdigit() or int(original) < 1 or str(canonical) not in labels:
                raise ValueError('association must use positive source and declared target labels')
            selected = source == int(original)
            if not selected.any():
                raise ValueError(f'source instance {original} is absent from {name}')
            target[selected] = int(canonical)
        filename = Path(name).with_suffix('.npy').name
        if filename in masks:
            raise ValueError('selected frames have colliding mask filenames')
        masks[filename] = target
        receipts[name] = {'source_mask_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'associations': mapping, 'pixels': {
                              str(k): int((target == int(k)).sum()) for k in labels}}
    chosen = {im.id: im for im in rec.images.values() if im.name in frames}
    # Validate everything before creating output; originals remain untouched.
    out.mkdir(parents=True)
    write_model(Reconstruction(rec.cameras, chosen, rec.xyz, rec.rgb), out/'sparse')
    (out/'masks').mkdir()
    for name, mask in masks.items():
        np.save(out/'masks'/name, mask)
    (out/'masks/labels.json').write_text(json.dumps(labels, indent=2)+'\n', encoding='utf-8')
    report = {'format':'playworld-instance-review/1', 'automatic_association':False,
              'scope':'Reviewed masks and selected cameras for CPU world assembly; training unchanged.',
              'review':review, 'frames':receipts,
              'source_sparse_sha256': {name:hashlib.sha256((scene/'sparse'/name).read_bytes()).hexdigest()
                                      for name in ['cameras.bin','images.bin','points3D.bin']}}
    (out/'instance-review.json').write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(remap_instances(a.scene, a.review, a.out), indent=2))
