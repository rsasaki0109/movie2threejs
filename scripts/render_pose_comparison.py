"""Evaluate both saved models from the same provided held-out cameras; no training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from playworld.colmap_io import read_model


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison-job', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    job = json.loads(a.comparison_job.read_text())
    if job['status'] != 'complete' or (a.out.exists() and any(a.out.iterdir())):
        raise ValueError('Use a completed experiment and a fresh render directory')
    os.environ['MPLBACKEND'] = 'Agg'
    import torch
    from gsplat import rasterization
    from torchmetrics.functional.image import structural_similarity_index_measure

    reference = a.comparison_job.parent/'inputs/provided'
    rec = read_model(reference/'sparse')
    images = rec.sorted_images()
    indices = list(range(0, len(images), 8))
    a.out.mkdir(parents=True)
    report = {'format': 'playworld-common-pose-render/1',
              'scope': 'Raw final Gaussian weights; identical dataset-provided held-out poses and pinhole intrinsics; black background, SH3, classic gsplat rasterizer',
              'comparison_job_sha256': digest(a.comparison_job),
              'renderer_script_sha256': digest(__file__),
              'held_out_sorted_indices': indices, 'torch': torch.__version__,
              'view_matrices': {}, 'variants': {}}
    targets = {}
    for i in indices:
        image = images[i]
        camera = rec.cameras[image.camera_id]
        view = np.eye(4)
        view[:3,:3], view[:3,3] = image.R, image.tvec
        target = np.array(Image.open(reference/'images'/image.name).convert('RGB'))
        if target.shape[:2] != (camera.height, camera.width):
            raise ValueError('Reference photograph and camera dimensions differ')
        Image.fromarray(target).save(a.out/f'reference_{i:02d}.png')
        targets[i] = target
        report['view_matrices'][str(i)] = {'image': image.name, 'world_to_camera': view.tolist(),
            'K': camera.K.tolist(), 'width': camera.width, 'height': camera.height,
            'photograph_sha256': digest(reference/'images'/image.name)}
    for name, run in job['runs'].items():
        checkpoint = Path(run['path'])/'scene/gs/ckpts/ckpt_6999_rank0.pt'
        stored = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if int(stored['step']) != 6999:
            raise ValueError('Expected exactly 7,000 completed training steps')
        splats = {k:v.cuda() for k,v in stored['splats'].items()}
        rows = []
        with torch.inference_mode():
            for i in indices:
                camera = report['view_matrices'][str(i)]
                started = time.perf_counter()
                colors, _, _ = rasterization(means=splats['means'], quats=splats['quats'],
                    scales=splats['scales'].exp(), opacities=splats['opacities'].sigmoid(),
                    colors=torch.cat([splats['sh0'], splats['shN']], 1),
                    viewmats=torch.tensor([camera['world_to_camera']],dtype=torch.float32,device='cuda'),
                    Ks=torch.tensor([camera['K']],dtype=torch.float32,device='cuda'),
                    width=camera['width'],height=camera['height'],sh_degree=3,
                    near_plane=0.01,far_plane=1e10,packed=False,rasterize_mode='classic',camera_model='pinhole')
                colors = colors.clamp(0,1)
                target = torch.tensor(targets[i],dtype=torch.float32,device='cuda')[None]/255
                mse = float((colors-target).square().mean().item())
                ssim = float(structural_similarity_index_measure(colors.permute(0,3,1,2),target.permute(0,3,1,2),data_range=1.0).item())
                pixels = (colors[0].cpu().numpy()*255).astype(np.uint8)
                output = a.out/f'{name}_{i:02d}.png'
                Image.fromarray(pixels).save(output)
                rows.append({'sorted_index':i,'image':camera['image'],'mse':mse,
                    'psnr_db':-10*math.log10(max(mse,1e-30)), 'ssim':ssim,
                    'seconds':time.perf_counter()-started,'png_sha256':digest(output)})
                print(name, i, rows[-1]['psnr_db'], rows[-1]['ssim'], flush=True)
        report['variants'][name] = {'checkpoint_sha256':digest(checkpoint),
            'gaussians':len(splats['means']),'views':rows,
            'mean_psnr_db':float(np.mean([r['psnr_db'] for r in rows])),
            'mean_ssim':float(np.mean([r['ssim'] for r in rows]))}
        del splats, stored, colors, target
        torch.cuda.empty_cache()
    (a.out/'common-view-report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
