"""Run measured public-data showcase stages on an already provisioned GPU host."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', choices=['apartment', 'office1b', 'kitchen', 'office_view2', 'raf_furnishedroom'])
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu-root', type=Path, default=ROOT / '.gpu')
    p.add_argument('--camera', default='19,16,22')
    p.add_argument('--frames', type=int, default=64)
    p.add_argument('--start-index', type=int, default=0)
    p.add_argument('--stop-index', type=int)
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--prompts', default='cardboard box,plastic bottle')
    p.add_argument('--seed-frame', type=int, default=5)
    p.add_argument('--bounds-margin', type=float, default=3)
    a = p.parse_args()
    scene, out, gpu = a.scene.resolve(), a.out.resolve(), a.gpu_root.resolve()
    if (scene.exists() and any(scene.iterdir())) or (out.exists() and any(out.iterdir())):
        raise FileExistsError('Choose fresh scene and output directories; saved runs are never overwritten')
    if a.steps < 1 or not 0 <= a.seed_frame < 16 or a.bounds_margin <= 0:
        raise ValueError('Steps and margin must be positive; seed-frame must be in [0, 15]')
    interpreters = {stage: gpu / f'envs/{stage}/bin/python' for stage in ['core', 'gsplat', 'sam3']}
    for interpreter in interpreters.values():
        if not interpreter.is_file():
            raise FileNotFoundError(f'Run scripts/setup_gpu.sh first: {interpreter}')
    preflight = "import torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability()[0] >= 8, 'SAM 3 needs an Ampere-or-newer GPU'; from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam3', 'config.json')"
    subprocess.run([str(interpreters['sam3']), '-c', preflight], check=True)
    scene.mkdir(parents=True, exist_ok=True)
    report = {'dataset': a.dataset, 'camera_source': 'provided calibration and sparse initialization; no VGGT',
              'status': 'running', 'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True).strip(), 'stages': {}}
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT / 'scripts'), PYTHONUNBUFFERED='1', MPLBACKEND='Agg')

    def stage(name, command):
        started = time.perf_counter()
        command = [str(item) for item in command]
        print('Starting', name, flush=True)
        try:
            with (scene / f'{name}.log').open('w') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            report['stages'][name] = {'seconds': round(time.perf_counter() - started, 3), 'exit_code': result.returncode, 'command': command}
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, command)
        except BaseException:
            report['status'] = 'interrupted-or-failed'
            report['interrupted_stage'] = name
            raise
        finally:
            (scene / 'calibrated-run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    core = interpreters['core']
    # Preparation needs an empty directory. Its report/log live next to the
    # downloaded scene, so creating logs cannot make that input nonempty.
    capture = scene / 'capture'
    prepare = [core, ROOT / 'scripts/prepare_calibrated_public.py', a.dataset, '--out', capture,
               '--frames', a.frames, '--camera', a.camera, '--start-index', a.start_index]
    if a.stop_index is not None:
        prepare += ['--stop-index', a.stop_index]
    stage('prepare', prepare)
    stage('train', [core, '-m', 'playworld.cli', 'train', capture, '--gsplat-dir', gpu / 'repos/gsplat',
                    '--python', interpreters['gsplat'], '--steps', a.steps])
    segment = capture / 'segment-scene'
    stage('segment', [core, '-m', 'playworld.cli', 'segment', segment, '--python', interpreters['sam3'],
                      '--prompts', a.prompts, '--seed-frame', a.seed_frame])
    sys.path.insert(0, str(ROOT / 'src'))
    from playworld.cli import latest_ply
    stage('world', [core, '-m', 'playworld.cli', 'world', segment, '--splats', latest_ply(capture),
                    '--out', out, '--bounds-margin', a.bounds_margin, '--clean-splats'])
    report['status'] = 'complete'
    report['stats'] = json.loads((out / 'world.json').read_text())['stats']
    (scene / 'calibrated-run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
