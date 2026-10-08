import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from playworld.pose_quality import fit_similarity, rotation_angle, transform_camera


def test_similarity_and_camera_preserve_projection_on_synthetic_room():
    rng=np.random.default_rng(7)
    centers=rng.uniform([-2,.8,-2],[2,1.8,2],(20,3))
    frame=Rotation.from_euler('xyz',[15,55,-8],degrees=True).as_matrix()
    offset=np.array([4,-2,7]); scale=3.7
    reference=scale*(centers@frame.T)+offset
    s,R,t=fit_similarity(centers,reference)
    np.testing.assert_allclose(s,scale)
    np.testing.assert_allclose(R,frame,atol=1e-12)
    np.testing.assert_allclose(t,offset,atol=1e-12)
    camera=Rotation.from_euler('xyz',[5,20,3],degrees=True).as_matrix()
    camera_t=-camera@centers[0]
    new_R,new_t=transform_camera(camera,camera_t,s,R,t)
    points=rng.uniform(-1,1,(30,3))+centers[0]+camera.T@np.array([0,0,4])
    before=points@camera.T+camera_t
    after=(scale*(points@frame.T)+offset)@new_R.T+new_t
    np.testing.assert_allclose(after,scale*before,atol=1e-12)
    np.testing.assert_allclose(-new_R.T@new_t,reference[0],atol=1e-12)


def test_rotation_comparison_detects_opposite_room_view_without_reflection():
    assert rotation_angle(np.eye(3))==0
    opposite=Rotation.from_euler('y',176,degrees=True).as_matrix()
    assert rotation_angle(opposite)==pytest.approx(176)
    with pytest.raises(ValueError,match='proper rotation'):
        rotation_angle(np.diag([1,1,-1]))


@pytest.mark.parametrize('points', [np.zeros((4,3)),np.column_stack([np.arange(4),np.zeros((4,2))]),np.full((4,3),np.nan)])
def test_degenerate_camera_paths_are_not_given_spurious_alignment(points):
    with pytest.raises(ValueError):
        fit_similarity(points,points)
