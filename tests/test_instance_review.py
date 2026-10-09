import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from playworld.colmap_io import read_model
from playworld.synthetic import make_scene

spec = importlib.util.spec_from_file_location('remap_instances', Path(__file__).parents[1]/'scripts/remap_instances.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def scene(tmp_path):
    source = tmp_path/'scene'
    make_scene(source, seed=3)
    rec = read_model(source/'sparse')
    names = [im.name for im in rec.sorted_images()
             if (np.load(source/'masks'/Path(im.name).with_suffix('.npy')) == 1).any()][:2]
    assert len(names) == 2
    # Simulate SAM tracking the same observed object under two different IDs.
    original = []
    for name, label in zip(names, [7, 9]):
        path = source/'masks'/Path(name).with_suffix('.npy')
        mask = np.load(path)
        foreground = mask == 1
        assert foreground.any()
        mask[foreground] = label
        np.save(path, mask)
        original.append((path, hashlib.sha256(path.read_bytes()).hexdigest()))
    review = {'labels':{'1':'crate'}, 'frames':dict(zip(names, [{'7':1}, {'9':1}]))}
    file = tmp_path/'review.json'
    file.write_text(json.dumps(review))
    return source, file, review, original


def test_review_associates_switched_ids_without_mutating_source(scene, tmp_path):
    source, file, review, original = scene
    out = tmp_path/'reviewed'
    receipt = module.remap_instances(source, file, out)
    rec = read_model(out/'sparse')
    assert {im.name for im in rec.images.values()} == set(review['frames'])
    np.testing.assert_array_equal(rec.xyz, read_model(source/'sparse').xyz)
    for path, digest in original:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        mask = np.load(out/'masks'/path.name)
        assert set(np.unique(mask)) == {0, 1}
    assert receipt['automatic_association'] is False
    with pytest.raises(FileExistsError):
        module.remap_instances(source, file, out)


@pytest.mark.parametrize('error', ['missing_frame', 'absent_instance', 'undeclared_label'])
def test_invalid_review_fails_before_writing_output(scene, tmp_path, error):
    source, file, review, original = scene
    first = next(iter(review['frames']))
    if error == 'missing_frame':
        review['frames']['not-in-capture.jpg'] = {'7':1}
    elif error == 'absent_instance':
        review['frames'][first] = {'999':1}
    else:
        review['frames'][first] = {'7':42}
    file.write_text(json.dumps(review))
    out = tmp_path/'rejected'
    with pytest.raises(ValueError):
        module.remap_instances(source, file, out)
    assert not out.exists()
    for path, digest in original:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
