"""Rebuild a reviewed demo with selected SAM instances; other masks stay static."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from playworld.world import build_world


def select(target: Path, keep: list[int]):
    original=target/'scene/masks'
    selected=target/'masks-reviewed'
    selected.mkdir(exist_ok=False)
    labels=json.loads((original/'labels.json').read_text())
    if any(str(i) not in labels for i in keep):
        raise ValueError('Requested ID is absent from original SAM labels')
    (selected/'labels.json').write_text(json.dumps({str(i):labels[str(i)] for i in keep},indent=2))
    for file in original.glob('*.npy'):
        mask=np.load(file)
        np.save(selected/file.name,np.where(np.isin(mask,keep),mask,0).astype(mask.dtype))
    plys=list((target/'scene/gs/ply').glob('*.ply'))
    started=time.perf_counter()
    world=build_world(target/'scene/sparse',max(plys,key=lambda p:p.stat().st_mtime),target/'world-reviewed',selected)
    report={'selection':'Manual browser review of reconstructed instances',
            'kept_instance_ids':keep,'original_sam_labels':labels,
            'world_seconds':round(time.perf_counter()-started,3),'stats':world['stats']}
    (target/'reviewed-assembly.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('target',type=Path)
    p.add_argument('--keep',type=int,nargs='+',required=True)
    a=p.parse_args()
    select(a.target,a.keep)
