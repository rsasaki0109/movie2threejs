"""Train a locally prepared calibrated showcase on an existing GPU; no allocation.

Keep the final checkpoint, masks and raw PLY so object selection can be reviewed
and rebuilt on a CPU after the runtime is stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

from notebook_support import NotebookRun

ROOT = Path(__file__).resolve().parents[1]


def extract_prepared(archive: Path, destination: Path, expected_sha256: str):
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected_sha256:
            raise ValueError('Input ZIP differs from the prepared photographs')
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError('Prepared scene must be a new directory')
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('Input ZIP CRC failure')
        members = zipped.infolist()
        if sum(item.file_size for item in members) > 2 * 1024**3:
            raise ValueError('Prepared archive unexpectedly large')
        names = {item.filename for item in members}
        required = {'calibrated-source.json', 'EYEFULTOWER-LICENSE.txt',
                    'sparse/cameras.bin', 'sparse/images.bin', 'sparse/points3D.bin',
                    'segment-scene/sparse/cameras.bin', 'segment-scene/sparse/images.bin',
                    'segment-scene/sparse/points3D.bin'}
        if not required.issubset(names):
            raise ValueError('Missing prepared calibration or license')
        for item in members:
            if not (destination / item.filename).resolve().is_relative_to(destination):
                raise ValueError('Archive path escaped prepared scene')
        if len(names) != len(members):
            raise ValueError('Duplicate archive member')
        report = json.loads(zipped.read('calibrated-source.json'))
        if report.get('vggt_used') is not False or report.get('license') != 'MIT':
            raise ValueError('Expected reviewed, dataset-calibrated photographs')
        for download in report['downloads']:
            relative = download['path']
            if not relative.startswith('images/'):
                continue
            content = zipped.read(relative)
            if len(content) != download['bytes'] or hashlib.sha256(content).hexdigest() != download['sha256']:
                raise ValueError('Photograph differs from its source receipt')
        training_images = {name for name in names if name.startswith('images/') and not name.endswith('/')}
        segment_images = {name for name in names if name.startswith('segment-scene/images/') and not name.endswith('/')}
        if len(training_images) != report['frames'] or len(segment_images) != report['segmentation_frames']:
            raise ValueError('Prepared photograph count differs from its receipt')
        for name in segment_images:
            if zipped.read(name) != zipped.read(name.removeprefix('segment-scene/')):
                raise ValueError('Segmentation photograph differs from training photograph')
        zipped.extractall(destination)
    return report


class PreparedRun(NotebookRun):
    def bundle(self):
        archive = super().bundle()
        if self.report['completed']:
            patterns = ('segment-scene/sparse/*.bin', 'segment-scene/masks/*.npy',
                        'segment-scene/masks/labels.json', 'gs/cfg.yml', 'gs/stats/*.json',
                        'gs/renders/val_step*.png')
            final_step = self.report['settings']['train_steps'] - 1
            files = [path for pattern in patterns for path in self.scene.glob(pattern) if path.is_file()]
            files.append(self.scene / 'gs/ply' / f'point_cloud_{final_step}.ply')
            with zipfile.ZipFile(archive, 'a', compression=zipfile.ZIP_DEFLATED) as zipped:
                for path in files:
                    if not path.is_file():
                        raise FileNotFoundError('Missing final raw PLY')
                    zipped.write(path, 'scene/' + path.relative_to(self.scene).as_posix())
        return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu-root', type=Path, default=ROOT / '.gpu')
    parser.add_argument('--steps', type=int, default=7000)
    parser.add_argument('--prompts', default='chair')
    parser.add_argument('--seed-frame', type=int, default=5)
    parser.add_argument('--bounds-margin', type=float, default=6)
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.seed_frame < 16 or args.bounds_margin <= 0:
        raise ValueError('Positive steps/margin and a seed frame in [0, 15] required')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Choose a fresh attempt directory')
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.pop('PYTHONPATH', None)
    os.environ['PYTHONUNBUFFERED'] = '1'
    gpu = args.gpu_root.resolve()
    python = {stage: gpu / f'envs/{stage}/bin/python' for stage in ('core', 'gsplat', 'sam3')}
    for interpreter in python.values():
        if not interpreter.is_file():
            raise FileNotFoundError(f'Run scripts/setup_gpu.sh first: {interpreter}')
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
    if free < 14000:
        raise RuntimeError('Less than 14 GB GPU memory free; no other job will be stopped')
    token_file = Path('/content/.playworld-hf-token')
    if not os.environ.get('HF_TOKEN') and token_file.is_file():
        token_file.chmod(0o600)
        token = token_file.read_text().strip()
        token_file.unlink()
        if not token or any(not 33 <= ord(char) <= 126 for char in token):
            raise ValueError('HF token format invalid')
        os.environ['HF_TOKEN'] = token
    subprocess.run([str(python['sam3']), '-c',
        "from colab_job import preflight; assert preflight(), 'SAM 3 preflight failed'"],
        cwd=ROOT / 'scripts', check=True)
    args.out.mkdir(parents=True, exist_ok=True)
    run = PreparedRun(args.out, ROOT, {'train_steps': args.steps,
        'input_sha256': args.expected_sha256, 'camera_source': 'dataset provided; no VGGT',
        'prompts': args.prompts, 'seed_frame': args.seed_frame, 'bounds_margin': args.bounds_margin})
    report = {'format': 'playworld-calibrated-prepared-run/1', 'status': 'preparing',
              'path': run.path.as_posix(), 'backup_index': (run.path / 'backup-index.json').as_posix(),
              'input_sha256': args.expected_sha256,
              'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                              '--format=csv,noheader'], text=True).strip()}
    report_path = args.out / 'prepared-job.json'
    def save():
        temporary = report_path.with_suffix('.json.partial')
        temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        temporary.replace(report_path)
    save()
    print('Prepared run:', json.dumps(report), flush=True)
    started = time.monotonic()
    try:
        report['preparation'] = extract_prepared(args.input, run.scene, args.expected_sha256)
        run.source.mkdir()
        for name in ('calibrated-source.json', 'EYEFULTOWER-LICENSE.txt'):
            shutil.copy2(run.scene / name, run.source / name)
        report['status'] = 'training'
        save()
        run.execute([str(python['core']), '-m', 'playworld.cli', 'train', str(run.scene),
            '--gsplat-dir', str(gpu / 'repos/gsplat'), '--python', str(python['gsplat']),
            '--steps', str(args.steps)], 'train')
        report['status'] = 'segmenting'
        save()
        segment = run.scene / 'segment-scene'
        run.execute([str(python['core']), '-m', 'playworld.cli', 'segment', str(segment),
            '--python', str(python['sam3']), '--prompts', args.prompts,
            '--seed-frame', str(args.seed_frame)], 'segment')
        report['status'] = 'assembling'
        save()
        ply = run.scene / 'gs/ply' / f'point_cloud_{args.steps-1}.ply'
        run.execute([str(python['core']), '-m', 'playworld.cli', 'world', str(segment),
            '--splats', str(ply), '--out', str(run.world), '--bounds-margin', str(args.bounds_margin),
            '--clean-splats'], 'pipeline')
        shutil.copy2(run.source / 'EYEFULTOWER-LICENSE.txt', run.world / 'EYEFULTOWER-LICENSE.txt')
        run.prepare_backups()
        report['status'] = 'complete'
        report['stats'] = json.loads((run.world / 'world.json').read_text())['stats']
    except BaseException as error:
        report['status'] = 'interrupted-or-failed'
        report['error_type'] = type(error).__name__
        raise
    finally:
        report['seconds'] = time.monotonic() - started
        report['stages'] = run.report['stages']
        save()


if __name__ == '__main__':
    main()
