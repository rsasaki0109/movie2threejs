"""Assemble a static gallery and standalone viewers from checked demo assets."""
import argparse
import json
import shutil
from pathlib import Path

from build_demo import ROOT, build_site


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'_site')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {args.out}')
    shutil.copytree(ROOT/'web/gallery', args.out, dirs_exist_ok=True)
    demos = json.loads((ROOT/'web/gallery/demos.json').read_text())
    for demo in demos:
        if not demo['id'].replace('-', '').isalnum():
            raise ValueError('Invalid demo ID')
        assets = ROOT/'demo-assets'/demo['id']
        manifest = json.loads((assets/'manifest.json').read_text())
        if manifest['objects'] != demo['objects']:
            raise ValueError(f"Stale object count: {demo['id']}")
        target = args.out/'demos'/demo['id']
        build_site(assets, target)
        # Each standalone viewer receives its own actual timeline and preview.
        shutil.copy2(ROOT/demo['shot'], target/'hero-shot.json')
        shutil.copy2(ROOT/demo['video'], target/'hero.mp4')
        shutil.copy2(ROOT/demo['gif'], target/'hero.gif')
        preview = args.out/'previews'/f"{demo['id']}.gif"
        preview.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/demo['gif'], preview)
    (args.out/'.nojekyll').touch()
    print(f"Gallery: {len(demos)} checked demos, {args.out}")


if __name__=='__main__':
    main()
