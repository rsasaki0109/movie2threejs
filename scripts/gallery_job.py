"""Run one gallery recipe in an existing Colab; never allocate a runtime.

Use an isolated workspace and the already checked playworld stage environments.
This script runs inside the existing core venv, with HF_TOKEN in its environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = 'https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/EyefulTower'
LICENSE_REVISION = '06a01a4915afc872b893c20a025a0e14598c8478'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_recipe(recipe, root, code, gpu, steps, resume_at=None):
    target = root / recipe['id']
    target.mkdir(exist_ok=bool(resume_at))
    url = f"{BASE}/{recipe['dataset']}/images-jpeg-2k/{recipe['camera']}.mp4"
    original, clip = target / 'source.mp4', target / 'input.mp4'
    if not original.exists():
        with urllib.request.urlopen(url, timeout=90) as response:
            original.write_bytes(response.read())
    notice = urllib.request.urlopen(f'https://raw.githubusercontent.com/facebookresearch/EyefulTower/{LICENSE_REVISION}/LICENSE', timeout=30).read()
    if not notice.startswith(b'MIT License'):
        raise ValueError('Expected dataset MIT notice')
    (target / 'EYEFULTOWER-LICENSE.txt').write_bytes(notice)
    if not clip.exists():
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-ss',str(recipe['start']),'-i',str(original),
                        '-t',str(recipe['duration']),'-map','0:v:0','-an','-c:v','libx264','-crf','15','-preset','fast',
                        '-pix_fmt','yuv420p',str(clip)], check=True)
    provenance = {**recipe, 'url':url, 'kind':'capture-rig photograph sequence visualization',
                  'license':'MIT', 'license_revision':LICENSE_REVISION,
                  'source_sha256':sha256(original), 'input_sha256':sha256(clip)}
    (target/'source.json').write_text(json.dumps(provenance,indent=2))
    environment = dict(os.environ, PYTHONPATH=str(code/'src'), MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    # Bound only our stage processes. Existing GPU jobs and the kernel are untouched.
    bootstrap = root/'memory_entry.py'
    bootstrap.write_text("import runpy,sys,subprocess,time\nfrom pathlib import Path\n"
                         "for attempt in range(180):\n"
                         "    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())\n"
                         "    if free >= 12000: break\n"
                         "    if attempt % 6 == 0: print(f'Waiting for 12000 MiB free VRAM; currently {free} MiB',flush=True)\n"
                         "    time.sleep(10)\n"
                         "else: raise RuntimeError('No free GPU capacity; existing jobs were not touched')\n"
                         "import torch\ntorch.cuda.set_per_process_memory_fraction(.48)\nsys.argv=sys.argv[1:]\n"
                         "sys.path.insert(0,str(Path(sys.argv[0]).resolve().parent))\nrunpy.run_path(sys.argv[0],run_name='__main__')\n")
    interpreters = {}
    for stage in ['vggt','gsplat','sam3']:
        wrapper = root/f'{stage}-python'
        wrapper.write_text(f'#!/bin/sh\nexec "{gpu}/envs/{stage}/bin/python" "{bootstrap}" "$@"\n')
        wrapper.chmod(0o700)
        interpreters[stage] = str(wrapper)
    command = [sys.executable,'-m','playworld.cli','all',str(clip),str(target/'scene'),'--out',str(target/'world'),
               '--vggt-dir',str(gpu/'repos/vggt'),'--gsplat-dir',str(gpu/'repos/gsplat'),
               '--vggt-python',interpreters['vggt'],'--gsplat-python',interpreters['gsplat'],'--sam3-python',interpreters['sam3'],
               '--num','16','--max-size','1024','--steps',str(steps),'--pose-confidence','1.5',
               '--ba','--shared-camera','--ba-track-budget','8192','--ba-reprojection-error','32.0',
               '--prompts',recipe['prompts'],'--report',str(target/(f'run-resume-{resume_at}.json' if resume_at else 'run.json'))]
    if resume_at:
        command += ['--start-at', resume_at]
    (target/('command-resume.json' if resume_at else 'command.json')).write_text(json.dumps(command,indent=2))
    print('GALLERY RECIPE:',json.dumps(recipe),flush=True)
    started = time.perf_counter()
    result = subprocess.run(command,env=environment)
    status = {'id':recipe['id'],'exit_code':result.returncode,'seconds':round(time.perf_counter()-started,3)}
    (target/('job-status-resume.json' if resume_at else 'job-status.json')).write_text(json.dumps(status,indent=2))
    print('GALLERY JOB STATUS:',json.dumps(status),flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,default=Path('/content/playworld/.gpu'))
    parser.add_argument('--steps',type=int,default=4500)
    parser.add_argument('--resume-at',choices=['poses','train','segment','world'])
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    recipes = json.loads((code/'shots/gallery/recipes.json').read_text())
    recipe = next(r for r in recipes if r['id']==args.recipe)
    args.root.mkdir(exist_ok=True,parents=True)
    raise SystemExit(run_recipe(recipe,args.root,code,args.gpu,args.steps,args.resume_at))


if __name__=='__main__':
    main()
