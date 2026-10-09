import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image as PillowImage
import pytest

from playworld.colmap_io import Camera, Image
from playworld.objects import View, project

spec = importlib.util.spec_from_file_location('refine_checkpoint', Path(__file__).parents[1]/'scripts/refine_checkpoint.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_resizing_uses_each_images_actual_size_and_its_own_intrinsics(tmp_path):
    (tmp_path/'images').mkdir()
    PillowImage.new('RGB',(80,120),'red').save(tmp_path/'images/source.png')
    camera = Camera(1,'PINHOLE',160,240,np.array([120,150,80,120]))
    image = Image(1,np.array([1,0,0,0]),np.array([0,0,1]),1,'source.png')
    pixels,K,V = module.load_view(tmp_path,image,camera,60)
    assert pixels.shape == (60,40,3)
    np.testing.assert_allclose(K,[[30,0,20],[0,37.5,30],[0,0,1]])
    np.testing.assert_allclose(V[:3,3],[0,0,1])


def test_crop_keeps_original_rays_without_rescaling_the_focal_length():
    height,width = 80,100
    yy,xx = np.indices((height,width))
    pixels = np.stack([xx,yy,np.zeros_like(xx)],axis=-1).astype(np.uint8)
    K = np.array([[70.,0,50],[0,80,40],[0,0,1]])
    cropped, new_K = module.crop_view(pixels,K,32,np.random.default_rng(5))
    x,y = cropped[0,0,:2]
    points = np.array([[.1,.2,1.],[.2,-.1,2.]])
    u,v,_ = project(points,View(K,np.eye(3),np.zeros(3),width,height))
    cu,cv,_ = project(points,View(new_K,np.eye(3),np.zeros(3),32,32))
    np.testing.assert_allclose(cu,u-x)
    np.testing.assert_allclose(cv,v-y)
    np.testing.assert_array_equal(K,[[70,0,50],[0,80,40],[0,0,1]])


def test_small_images_crop_without_upscaling():
    pixels=np.zeros((20,15,3),dtype=np.uint8)
    K=np.eye(3)
    cropped,new_K=module.crop_view(pixels,K,512,np.random.default_rng(0))
    assert cropped.shape == pixels.shape
    np.testing.assert_array_equal(new_K,K)


def test_distorted_images_are_not_silently_treated_as_pinhole(tmp_path):
    camera=Camera(1,'OPENCV',100,100,np.zeros(8))
    image=Image(1,np.array([1,0,0,0]),np.zeros(3),1,'source.jpg')
    with pytest.raises(ValueError,match='undistorted'):
        module.load_view(tmp_path,image,camera,100)


def test_ssim_detects_loss_of_detail_and_allows_backward():
    torch=pytest.importorskip('torch')
    torch.manual_seed(0)
    image=torch.rand((1,3,32,32))
    assert float(module.ssim(image,image)) == pytest.approx(1,abs=1e-5)
    blurred=torch.nn.functional.avg_pool2d(image,5,stride=1,padding=2).requires_grad_()
    loss=1-module.ssim(blurred,image)
    assert float(loss.detach())>.2
    loss.backward()
    assert torch.isfinite(blurred.grad).all()


def test_depth_prior_rejects_hidden_tracks_and_color_mismatch():
    K=np.array([[100.,0,50],[0,100,50],[0,0,1]])
    xyz=np.array([[0,0,1],[0,0,3],[.2,0,1],[0,.2,1]])
    rgb=np.array([[255,0,0],[255,0,0],[0,255,0],[0,0,255]],dtype=np.uint8)
    photo=np.zeros((100,100,3),dtype=np.uint8)
    photo[50,50]=[255,0,0]
    photo[70,50]=[0,0,255]
    uv,z=module.depth_samples(xyz,rgb,photo,K,np.eye(4))
    np.testing.assert_array_equal(uv,[[50,50],[50,70]])
    np.testing.assert_array_equal(z,[1,1])


def test_depth_sampling_is_bounded_and_reproducible():
    K=np.array([[100.,0,50],[0,100,50],[0,0,1]])
    xx,yy=np.meshgrid(np.arange(-.4,.41,.1),np.arange(-.4,.41,.1))
    xyz=np.column_stack([xx.ravel(),yy.ravel(),np.ones(xx.size)])
    rgb=np.full((len(xyz),3),100,dtype=np.uint8)
    photo=np.full((100,100,3),100,dtype=np.uint8)
    a=module.depth_samples(xyz,rgb,photo,K,np.eye(4),limit=8,seed=4)
    b=module.depth_samples(xyz,rgb,photo,K,np.eye(4),limit=8,seed=4)
    assert len(a[0]) == 8
    np.testing.assert_array_equal(a[0],b[0])
    np.testing.assert_array_equal(a[1],b[1])
