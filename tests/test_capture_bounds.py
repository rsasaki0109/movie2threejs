import numpy as np
import pytest
from playworld import colliders, gravity, splat_io
from playworld.world import build_world


def test_remote_outliers_do_not_coarsen_room_walls():
    y,z=np.meshgrid(np.linspace(.2,2.5,90),np.linspace(-2,2,120))
    wall=np.column_stack([np.zeros(y.size),y.ravel(),z.ravel()])
    points=np.vstack([wall,[[1000,2,1000],[-1000,2,-1000],[np.nan,0,0]]])
    keep=colliders.near_capture(points,np.array([[1,1.5,-1],[1,1.5,1]]),4)
    assert keep[:len(wall)].all() and not keep[-3:].any()
    boxes=colliders.static_boxes(points[keep])
    assert boxes and all(np.max(b.half_extents)<3 for b in boxes)
    for z in [-1.5,0,1.5]:
        assert any(np.all(np.abs(np.array([0,1,z])-b.center)<=b.half_extents+.001) for b in boxes)


def test_bounded_synthetic_room_keeps_objects_and_discards_remote_splat(tmp_path):
    from playworld.synthetic import make_scene
    root=tmp_path/'capture';make_scene(root,seed=3)
    source=root/'splats.ply';props=splat_io.read_ply(source)
    # A splat far away in reconstruction coordinates must not enlarge collision bounds.
    extra={k:np.append(v,v[0]) for k,v in props.items()}
    for name in ['x','y','z']:extra[name][-1]=10000
    splat_io.write_ply(source,extra)
    world=build_world(root/'sparse',source,tmp_path/'out',root/'masks',bounds_margin=6)
    assert sorted(o['name'] for o in world['objects'])==['chair','crate','mug']
    assert world['stats']['discarded_bounds_gaussians']==1
    T=np.array(world['align']).reshape(4,4).T
    for relative in [world['background'],*(o['splat'] for o in world['objects'])]:
        points=gravity.apply(T,splat_io.means(splat_io.read_ply(tmp_path/'out'/relative)))
        assert np.max(np.abs(points))<20


@pytest.mark.parametrize('margin',[0,-1,float('inf'),float('nan')])
def test_invalid_capture_margin_is_rejected(margin):
    with pytest.raises(ValueError):colliders.near_capture(np.zeros((1,3)),np.zeros((1,3)),margin)
