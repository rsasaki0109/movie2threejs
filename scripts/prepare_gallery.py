"""Create a provisional camera shot and compact export; browser QA is still required."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from build_demo import package_world
from playworld.colmap_io import read_model
from playworld.gravity import apply
from playworld.world import install_viewer


def prepare(target: Path):
    world = json.loads((target/'world/world.json').read_text())
    source = json.loads((target/'source.json').read_text())
    transform = np.array(world['align']).reshape(4,4).T
    reconstruction = read_model(target/'scene/sparse')
    images = reconstruction.sorted_images()
    centers = apply(transform, np.array([im.center for im in images]))
    directions = np.array([transform[:3,:3] @ im.R.T[:,2] for im in images])
    directions /= np.linalg.norm(directions,axis=1,keepdims=True)
    candidates = []
    for obj in world['objects']:
        center = np.array(obj['centroid'])
        delta = center - centers
        distances = np.linalg.norm(delta,axis=1)
        visible = np.sum(delta * directions,axis=1) / np.maximum(distances,1e-9)
        # Prefer a visible floor object; stay in a captured viewpoint.
        scores = visible - .07 * distances + (.3 if obj['support_y'] < .15 else 0)
        scores[(distances < .8) | (distances > 5)] -= 2
        index = int(scores.argmax())
        candidates.append((float(scores[index]),index,obj))
    if not candidates:
        raise ValueError('No movable objects: do not list this as a physics demo')
    _, index, obj = max(candidates,key=lambda item:item[0])
    position = centers[index].copy(); position[1] = 1.53
    target_point = np.array(obj['centroid'])
    target_point[1] += .1
    # A short camera move along the photographed trajectory, then back to loop.
    end_index = min(len(centers)-1,index+2) if index+2<len(centers) else max(0,index-2)
    near = position + (centers[end_index] - centers[index]) * .35
    near[1] = 1.53
    hull = np.array(obj['hull'])
    toward_camera = position - np.array(obj['centroid']); toward_camera[1] = 0
    toward_camera /= max(np.linalg.norm(toward_camera),1e-9)
    radius = float(np.linalg.norm(hull[:,[0,2]],axis=1).max())
    ray_target = np.array(obj['centroid'])
    ray_origin = ray_target + toward_camera * (radius + .25)
    action = {'type':'push','time':3,'origin':ray_origin.round(5).tolist(),
              'target':ray_target.round(5).tolist(),'strength':3}
    profile = {'player':{'spawn':[round(position[0],5),0,round(position[2],5)],
                         'look_at':target_point.round(5).tolist(),'fov':60},
               'demo':{'id':source['id'],'title':source['title'],
                       'description':f"Walk through a real {source['dataset']} capture. Push the {obj['name']} and throw a ball.",
                       'dataset':source['dataset'],'subject_id':obj['id'],
                       'push_label':f"Push the {obj['name']}",'push':{k:v for k,v in action.items() if k not in ['type','time']}}}
    shot = {'version':1,'fps':15,'duration':8,'fov':60,'camera':[
        {'time':0,'position':position.round(5).tolist(),'target':target_point.round(5).tolist()},
        {'time':2,'position':near.round(5).tolist(),'target':target_point.round(5).tolist()},
        {'time':6,'position':near.round(5).tolist(),'target':target_point.round(5).tolist()},
        {'time':8,'position':position.round(5).tolist(),'target':target_point.round(5).tolist()}],
        'events':[action,{'time':5.5,'type':'throw'}]}
    (target/'shot.json').write_text(json.dumps(shot,indent=2))
    (target/'profile.json').write_text(json.dumps(profile,indent=2))
    compact = target/'compact'
    package_world(target/'world',compact,profile)
    install_viewer(compact)
    for file in ['source.json','input.mp4','EYEFULTOWER-LICENSE.txt','shot.json','profile.json']:
        shutil.copy2(target/file, compact/file)
    for report in target.glob('run*.json'):
        shutil.copy2(report,compact/report.name)
    for file in ['pose-diagnostics.json']:
        if (target/'scene'/file).exists(): shutil.copy2(target/'scene'/file,compact/file)
    archive = Path(shutil.make_archive(str(target/'compact-result'),'zip',root_dir=compact))
    pieces=[]
    with archive.open('rb') as stream:
        for index in range(100):
            block=stream.read(16*1024*1024)
            if not block: break
            part=archive.with_suffix(f'.zip.part{index:02d}')
            part.write_bytes(block)
            pieces.append({'path':str(part),'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()})
    result={'id':source['id'],'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'pieces':pieces,'objects':len(world['objects']),'colliders':len(world['colliders']),
            'subject_id':obj['id'],'provisional':True}
    (target/'export.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target',type=Path)
    prepare(parser.parse_args().target)
