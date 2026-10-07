"""Preserve completed gallery work for transfer to another existing Colab GPU.

Only public inputs, extracted images, sparse poses, latest PLY and measured
reports are included. Environments, model weights and credentials are excluded.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    recipes=json.loads((root/'code/shots/gallery/recipes.json').read_text())
    archive=root/'gallery-resume.zip'
    entries=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as bundle:
        for recipe in recipes:
            target=root/recipe['id']
            if not target.is_dir(): continue
            files=[p for name in ['source.mp4','input.mp4','source.json','EYEFULTOWER-LICENSE.txt','command.json','command-resume.json']
                   if (p:=target/name).is_file()]
            files+=list(target.glob('run*.json'))+list((target/'scene/images').glob('*.png'))
            files+=list((target/'scene/sparse').rglob('*.bin'))
            plys=list((target/'scene/gs/ply').glob('*.ply'))
            if plys: files.append(max(plys,key=lambda p:p.stat().st_mtime))
            for file in sorted(set(files)):
                if not file.resolve().is_relative_to(root): raise ValueError('Unexpected migration path')
                bundle.write(file,file.relative_to(root).as_posix())
            entries.append({'id':recipe['id'],'resume_at':'segment' if plys else 'poses',
                            'files':len(set(files))})
    parts=[]
    with archive.open('rb') as stream:
        for index in range(1000):
            block=stream.read(16*1024*1024)
            if not block: break
            part=archive.with_suffix(f'.zip.part{index:02d}')
            part.write_bytes(block)
            parts.append({'path':str(part),'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()})
    report={'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'entries':entries,'parts':parts}
    (root/'gallery-resume-manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
