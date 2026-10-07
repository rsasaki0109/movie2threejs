"""Prepare an explicitly calibrated Eyeful Tower showcase, not a video pose benchmark."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from playworld.colmap_io import Camera, Reconstruction, read_model, write_model

BASE = 'https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/EyefulTower'
REVISION = '06a01a4915afc872b893c20a025a0e14598c8478'


def prepare(dataset: str, out: Path, frames: int = 64, camera: str = '19') -> dict:
    if dataset not in {'office1b', 'kitchen', 'office_view2', 'raf_furnishedroom', 'apartment'}:
        raise ValueError('Choose a reviewed MIT-licensed public dataset')
    if frames < 16:
        raise ValueError('At least 16 images required for this showcase recipe')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    downloads = []

    def download(relative: str, target: Path):
        if not target.resolve().is_relative_to(out.resolve()):
            raise ValueError('Dataset path escaped the output directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'{BASE}/{dataset}/{relative}'
        with urllib.request.urlopen(url, timeout=120) as response, target.open('wb') as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        record = {'url': url, 'path': target.relative_to(out).as_posix(),
                  'bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        return record

    for name in ['cameras.bin', 'images.bin', 'points3D.bin']:
        downloads.append(download(f'colmap/sparse/0/{name}', out / '_upstream/sparse' / name))
    rec = read_model(out / '_upstream/sparse')
    camera_names = camera.split(',')
    if not all(name.isdigit() for name in camera_names) or len(set(camera_names)) != len(camera_names):
        raise ValueError('Camera names must be unique numeric IDs, separated by commas')
    chosen = []
    primary = []
    for name in camera_names:
        candidates = [im for im in rec.sorted_images() if Path(im.name).name.startswith(name + '_')]
        if len(candidates) < 16:
            raise ValueError(f'Camera {name} has fewer than sixteen photographs')
        selected = [candidates[i] for i in np.unique(np.linspace(0, len(candidates)-1, min(frames, len(candidates))).round().astype(int))]
        chosen.extend(selected)
        if name == camera_names[0]:
            primary = selected
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download, 'colmap/images_8/' + im.name, out / 'images' / im.name) for im in chosen]
        # Preserve manifest order regardless of concurrent download completion.
        downloads.extend(f.result() for f in futures)
    cameras = {}
    for im in chosen:
        if Path(im.name).is_absolute() or '..' in Path(im.name).parts:
            raise ValueError('Unsafe dataset image name')
        cam = rec.cameras[im.camera_id]
        if cam.model != 'PINHOLE':
            raise ValueError('Expected the dataset-provided undistorted PINHOLE reconstruction')
        width, height = PILImage.open(out / 'images' / im.name).size
        params = cam.params * np.array([width / cam.width, height / cam.height,
                                       width / cam.width, height / cam.height])
        scaled = Camera(cam.id, cam.model, width, height, params)
        previous = cameras.get(cam.id)
        if previous is not None and (previous.width != width or previous.height != height):
            raise ValueError('Images sharing a camera must have identical dimensions')
        cameras[cam.id] = scaled
    selected_points = np.random.default_rng(42).choice(len(rec.xyz), min(100_000, len(rec.xyz)), replace=False)
    model = Reconstruction(cameras, {im.id: im for im in chosen}, rec.xyz[selected_points], rec.rgb[selected_points])
    write_model(model, out / 'sparse')
    # SAM 3 runs on sixteen of the same photographs. Its smaller reconstruction
    # uses the exact same world coordinates as the full training scene.
    segment = out / 'segment-scene'; (segment / 'images').mkdir(parents=True)
    subset = [primary[i] for i in np.linspace(0, len(primary)-1, 16).round().astype(int)]
    for im in subset:
        target = segment / 'images' / im.name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((out / 'images' / im.name).read_bytes())
    write_model(Reconstruction(cameras, {im.id: im for im in subset}, model.xyz, model.rgb), segment / 'sparse')
    report = {'dataset': dataset, 'camera': camera, 'primary_camera': camera_names[0],
              'frames_per_camera_requested': frames, 'frames': len(chosen), 'segmentation_frames': len(subset),
              'initial_points': len(model.xyz), 'camera_source': 'dataset-provided Metashape COLMAP calibration',
              'images': 'dataset-provided undistorted colmap/images_8 JPEGs',
              'vggt_used': False, 'license': 'MIT', 'license_revision': REVISION, 'downloads': downloads}
    (out / 'calibrated-source.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=64)
    parser.add_argument('--camera', default='19', help='one ID or comma-separated IDs; first camera supplies SAM views')
    args = parser.parse_args()
    result = prepare(args.dataset, args.out, args.frames, args.camera)
    print(json.dumps({k: v for k, v in result.items() if k != 'downloads'}, indent=2))
