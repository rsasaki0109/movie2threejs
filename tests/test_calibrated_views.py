import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_calibrated_public import select_views


def test_multicamera_interval_and_sampling_are_independent_of_input_order():
    images = [SimpleNamespace(name=f'{camera}_DSC{i:04d}.jpg')
              for camera in [19, 16, 22] for i in range(80)]
    for camera in ['19', '16', '22']:
        chosen = select_views(reversed(images), camera, 16, 30, 65)
        assert len(chosen) == 16
        assert chosen[0].name == f'{camera}_DSC0030.jpg'
        assert chosen[-1].name == f'{camera}_DSC0064.jpg'
        assert len({im.name for im in chosen}) == 16


def test_requesting_all_views_retains_every_selected_photograph():
    images = [SimpleNamespace(name=f'19_{i:04d}.jpg') for i in range(60)]
    assert len(select_views(images, '19', 64, 20, 52)) == 32


@pytest.mark.parametrize('start,stop', [(-1, 30), (10, 10), (0, 99), (40, 50)])
def test_invalid_or_insufficient_intervals_fail(start, stop):
    images = [SimpleNamespace(name=f'19_{i:04d}.jpg') for i in range(60)]
    with pytest.raises(ValueError):
        select_views(images, '19', 32, start, stop)
