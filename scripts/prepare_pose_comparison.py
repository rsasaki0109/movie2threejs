"""Prepare a controlled extrinsics comparison; never allocates a GPU or trains."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image as PILImage

from playworld.colmap_io import Image, Reconstruction, read_model, rotmat_to_qvec, write_model
from playworld.pose_quality import fit_similarity, rotation_angle, transform_camera


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(estimated_scene: Path, reference_scene: Path, mapping: Path, out: Path):
    estimated_scene, reference_scene, mapping, out = map(Path, (estimated_scene, reference_scene, mapping, out))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a fresh comparison directory')
    estimated, reference = read_model(estimated_scene/'sparse'), read_model(reference_scene/'sparse')
    entries = json.loads(mapping.read_text(encoding='utf-8'))
    estimate_by_name = {im.name: im for im in estimated.images.values()}
    reference_by_name = {im.name: im for im in reference.images.values()}
    if len(entries) < 3 or len({e['estimated_name'] for e in entries}) != len(entries) or len({e['reference_name'] for e in entries}) != len(entries):
        raise ValueError('Provide at least three distinct explicit image correspondences')
    entries = sorted(entries, key=lambda e:e['estimated_name'])
    a = [estimate_by_name[e['estimated_name']] for e in entries]
    b = [reference_by_name[e['reference_name']] for e in entries]
    paths=[]
    for image in b:
        path=(reference_scene/'images'/image.name).resolve()
        if not path.is_relative_to((reference_scene/'images').resolve()) or not path.is_file():
            raise ValueError('Missing or unsafe reference image')
        camera=reference.cameras[image.camera_id]
        if PILImage.open(path).size != (camera.width,camera.height):
            raise ValueError('Reference intrinsics do not match image dimensions')
        paths.append(path)
    scale, rotation, translation = fit_similarity(np.stack([im.center for im in a]),np.stack([im.center for im in b]))
    rows=[]
    models={name:{} for name in ('estimated','provided')}
    for image_id,(entry,est,ref) in enumerate(zip(entries,a,b),1):
        R,t = transform_camera(est.R,est.tvec,scale,rotation,translation)
        models['estimated'][image_id] = Image(image_id,rotmat_to_qvec(R),t,ref.camera_id,ref.name)
        models['provided'][image_id] = Image(image_id,ref.qvec.copy(),ref.tvec.copy(),ref.camera_id,ref.name)
        rows.append({**entry, 'center_residual_reference_units':float(np.linalg.norm(scale*rotation@est.center+translation-ref.center)),
                     'orientation_residual_degrees':rotation_angle(ref.R@R.T)})
    camera_ids={im.camera_id for im in b}
    cameras={cid:reference.cameras[cid] for cid in camera_ids}
    out.mkdir(parents=True,exist_ok=True)
    image_hashes={im.name:sha256(path) for im,path in zip(b,paths)}
    for name,images in models.items():
        scene=out/name
        (scene/'images').mkdir(parents=True)
        for image,path in zip(b,paths):
            shutil.copy2(path,scene/'images'/image.name)
        write_model(Reconstruction(cameras,images,reference.xyz,reference.rgb),scene/'sparse')
        if (reference_scene/'EYEFULTOWER-LICENSE.txt').is_file():
            shutil.copy2(reference_scene/'EYEFULTOWER-LICENSE.txt',scene/'EYEFULTOWER-LICENSE.txt')
    for name in ('cameras.bin','points3D.bin'):
        assert sha256(out/'estimated/sparse'/name)==sha256(out/'provided/sparse'/name)
    report={'format':'playworld-pose-comparison-input/1','new_training':False,
        'scope':'extrinsics comparison using identical provided undistorted photographs, intrinsics and sparse initialization; not a video first run',
        'mapping_sha256':sha256(mapping),'pairs':rows,'image_sha256':image_hashes,
        'shared_points':len(reference.xyz),'shared_intrinsics':True,'shared_initialization':True,
        'estimated_poses_transformed_by_one_global_similarity':True,
        'alignment':{'scale':scale,'rotation':rotation.tolist(),'translation':translation.tolist(),'fit_pairs':len(rows)},
        'center_rmse_reference_units':float(np.sqrt(np.mean([e['center_residual_reference_units']**2 for e in rows]))),
        'median_orientation_degrees':float(np.median([e['orientation_residual_degrees'] for e in rows])),
        'relative_rotations':[{'from':a[i].name,'to':a[i+1].name,
            'estimated_degrees':rotation_angle(a[i].R@a[i+1].R.T),
            'provided_degrees':rotation_angle(b[i].R@b[i+1].R.T),
            'residual_degrees':rotation_angle((a[i].R@a[i+1].R.T).T@(b[i].R@b[i+1].R.T))} for i in range(len(a)-1)],
        'source_model_sha256':{kind:{n:sha256(scene/'sparse'/n) for n in ('cameras.bin','images.bin','points3D.bin')}
                               for kind,scene in [('estimated',estimated_scene),('provided',reference_scene)]}}
    (out/'comparison-input.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('estimated-scene','reference-scene','mapping','out'):
        parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args()
    report=prepare(args.estimated_scene,args.reference_scene,args.mapping,args.out)
    print(json.dumps({k:report[k] for k in ('scope','shared_points','center_rmse_reference_units','median_orientation_degrees')},indent=2))
