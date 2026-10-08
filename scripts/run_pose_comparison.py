"""Train two prepared extrinsics variants on an already provisioned GPU; no allocation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

from notebook_support import NotebookRun

ROOT=Path(__file__).resolve().parents[1]


def train_command(core, scene, gpu, steps):
    return [str(core),'-m','playworld.cli','train',str(scene),'--gsplat-dir',str(gpu/'repos/gsplat'),
            '--python',str(gpu/'envs/gsplat/bin/python'),'--steps',str(steps)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu-root',type=Path,default=ROOT/'.gpu')
    p.add_argument('--steps',type=int,default=7000)
    a=p.parse_args()
    if a.steps<1 or (a.out.exists() and any(a.out.iterdir())):
        raise ValueError('Use positive steps and a fresh output directory')
    with a.input.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=a.expected_sha256:
            raise ValueError('Input ZIP differs from the prepared comparison')
    gpu=a.gpu_root.resolve(); core=gpu/'envs/core/bin/python'
    for executable in (core,gpu/'envs/gsplat/bin/python'):
        if not executable.is_file():
            raise FileNotFoundError(f'Run scripts/setup_gpu.sh first: {executable}')
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<14000:
        raise RuntimeError('Less than 14 GB GPU memory free; no other job will be stopped')
    a.out.mkdir(parents=True,exist_ok=True)
    inputs=(a.out/'inputs').resolve()
    with zipfile.ZipFile(a.input) as z:
        if z.testzip() is not None:
            raise ValueError('Input ZIP CRC failure')
        if sum(i.file_size for i in z.infolist())>2*1024**3:
            raise ValueError('Comparison archive unexpectedly large')
        for item in z.infolist():
            if not (inputs/item.filename).resolve().is_relative_to(inputs):
                raise ValueError('Archive path escaped comparison directory')
        z.extractall(inputs)
    preparation=json.loads((inputs/'comparison-input.json').read_text())
    def digest(path):
        with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    for name in ('cameras.bin','points3D.bin'):
        if digest(inputs/'estimated/sparse'/name)!=digest(inputs/'provided/sparse'/name):
            raise ValueError('Intrinsics and initialization must be identical')
    for variant in ('estimated','provided'):
        for relative,expected in preparation['image_sha256'].items():
            path=(inputs/variant/'images'/relative).resolve()
            if not path.is_relative_to((inputs/variant/'images').resolve()) or digest(path)!=expected:
                raise ValueError('Photographs differ from the prepared experiment')
    runs={name:NotebookRun(a.out,ROOT,{'train_steps':a.steps,'variant':name,
        'comparison_input_sha256':a.expected_sha256,'camera_source':'previous VGGT estimate' if name=='estimated' else 'dataset provided',
        'purpose':'controlled extrinsics quality comparison; static scene, no segmentation'}) for name in ('estimated','provided')}
    report={'format':'playworld-pose-comparison-run/1','status':'running','gpu':subprocess.check_output(
        ['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
        'input_sha256':a.expected_sha256,'preparation':preparation,
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'runs':{name:{'path':run.path.as_posix(),'backup_index':(run.path/'backup-index.json').as_posix()} for name,run in runs.items()}}
    report_path=a.out/'comparison-job.json'
    def save():report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    save(); print('Comparison paths:',json.dumps(report['runs']),flush=True)
    started=time.monotonic()
    try:
        for name,run in runs.items():
            shutil.copytree(inputs/name,run.scene)
            run.execute(train_command(core,run.scene,gpu,a.steps),'train')
            ply=run.scene/'gs/ply'/f'point_cloud_{a.steps-1}.ply'
            if not ply.is_file():
                raise FileNotFoundError('Expected final PLY for this exact training step')
            run.execute([str(core),'-m','playworld.cli','world',str(run.scene),
                '--splats',str(ply),'--out',str(run.world)],'pipeline')
            run.prepare_backups()
            report['runs'][name]['completed']=True
            report['runs'][name]['stats']=json.loads((run.world/'world.json').read_text())['stats']
            report['runs'][name]['stages']=run.report['stages']
            save()
        report['status']='complete'
    except BaseException as error:
        report['status']='interrupted-or-failed';report['error_type']=type(error).__name__
        raise
    finally:
        report['seconds']=time.monotonic()-started;save()


if __name__=='__main__':main()
