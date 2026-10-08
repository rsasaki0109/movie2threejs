import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_pose_comparison import prepare, sha256
from run_pose_comparison import train_command
from playworld.colmap_io import Camera, Image, Reconstruction, read_model, rotmat_to_qvec, write_model
from playworld.pose_quality import transform_camera
from playworld.pose_quality import rotation_angle


@pytest.mark.parametrize('flip_first',[False,True])
def test_prepared_variants_change_only_extrinsics_in_synthetic_room(tmp_path,flip_first):
    source,reference=tmp_path/'source',tmp_path/'reference'
    camera=Camera(1,'PINHOLE',64,48,np.array([50,50,32,24]))
    centers=np.array([[0,1,0],[1,1,0],[1,1,1],[0,1,1]],dtype=float)
    frame=Rotation.from_euler('y',35,degrees=True).as_matrix()
    offset=np.array([5,-1,3]);scale=2
    original={};provided={}
    for index,center in enumerate(centers,1):
        name=f'view-{index}.png'
        R=np.eye(3);t=-center
        source_R=Rotation.from_euler('y',180,degrees=True).as_matrix() if flip_first and index==1 else R
        original[index]=Image(index,rotmat_to_qvec(source_R),-source_R@center,1,name)
        RR,tt=transform_camera(R,t,scale,frame,offset)
        provided[index]=Image(index,rotmat_to_qvec(RR),tt,1,name)
        (reference/'images').mkdir(exist_ok=True,parents=True)
        PILImage.new('RGB',(64,48),(index*40,80,120)).save(reference/'images'/name)
    xyz=np.array([[0,0,3],[1,1,3],[1,0,2]],dtype=float);rgb=np.full((3,3),100,dtype=np.uint8)
    write_model(Reconstruction({1:camera},original,xyz,rgb),source/'sparse')
    write_model(Reconstruction({1:camera},provided,scale*(xyz@frame.T)+offset,rgb),reference/'sparse')
    mapping=tmp_path/'mapping.json'
    mapping.write_text(json.dumps([{'estimated_name':im.name,'reference_name':im.name} for im in original.values()]))
    source_hash=sha256(source/'sparse/images.bin')
    report=prepare(source,reference,mapping,tmp_path/'out')
    assert report['center_rmse_reference_units']<1e-12
    a=read_model(tmp_path/'out/estimated/sparse');b=read_model(tmp_path/'out/provided/sparse')
    for index in a.images:
        if flip_first and index==1:
            assert rotation_angle(a.images[index].R@b.images[index].R.T)==pytest.approx(180)
        else:
            np.testing.assert_allclose(a.images[index].R,b.images[index].R,atol=1e-12)
        np.testing.assert_allclose(a.images[index].center,b.images[index].center,atol=1e-12)
        name=a.images[index].name
        assert sha256(tmp_path/'out/estimated/images'/name)==sha256(tmp_path/'out/provided/images'/name)
    for name in ('cameras.bin','points3D.bin'):
        assert sha256(tmp_path/'out/estimated/sparse'/name)==sha256(tmp_path/'out/provided/sparse'/name)
    assert sha256(source/'sparse/images.bin')==source_hash


def test_comparison_uses_identical_training_options_for_both_variants():
    a=train_command(Path('/gpu/core/python'),Path('/runs/estimated'),Path('/gpu'),7000)
    b=train_command(Path('/gpu/core/python'),Path('/runs/provided'),Path('/gpu'),7000)
    assert a[:4]==b[:4] and a[5:]==b[5:]
    assert a[4]==str(Path('/runs/estimated')) and b[4]==str(Path('/runs/provided'))
