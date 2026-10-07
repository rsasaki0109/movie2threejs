import numpy as np
from playworld import splat_io


def test_horizontal_gaussian_normals_respect_world_alignment():
    props = splat_io.make_splats(np.zeros((3, 3)), np.ones((3, 3)) * .5, .1)
    props['scale_0'] = np.log([.1, .001, .1])
    props['scale_1'] = np.log([.001, .1, .1])
    props['scale_2'] = np.log([.1, .1, .1])
    assert splat_io.horizontal_mask(props, np.eye(4)).tolist() == [True, False, False]
    alignment = np.eye(4)
    alignment[:3, :3] = [[0, -2, 0], [2, 0, 0], [0, 0, 2]]
    assert splat_io.horizontal_mask(props, alignment).tolist() == [False, True, False]
    # Rotating the first thin splat 90 degrees about z makes it a wall.
    props['rot_0'][0] = np.sqrt(.5)
    props['rot_3'][0] = np.sqrt(.5)
    assert not splat_io.horizontal_mask(props, np.eye(4))[0]
