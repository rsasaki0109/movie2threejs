import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest


@pytest.fixture
def prepared(monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('calibrated_prepared', scripts / 'run_calibrated_prepared.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def input_archive(tmp_path, **changes):
    image = b'fixture photograph'
    report = {'vggt_used': False, 'license': 'MIT', 'frames': 1, 'segmentation_frames': 1,
              'downloads': [{'path': 'images/19_001.jpg', 'bytes': len(image),
                             'sha256': hashlib.sha256(image).hexdigest()}]}
    entries = {'calibrated-source.json': json.dumps(report).encode(),
               'EYEFULTOWER-LICENSE.txt': b'fixture license',
               'images/19_001.jpg': image, 'segment-scene/images/19_001.jpg': image}
    for prefix in ('sparse', 'segment-scene/sparse'):
        for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
            entries[f'{prefix}/{name}'] = b'fixture model'
    entries.update(changes)
    path = tmp_path / 'input.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_verified_photographs_extract_into_fresh_scene(prepared, tmp_path):
    archive, digest = input_archive(tmp_path)
    destination = tmp_path / 'scene'
    assert prepared.extract_prepared(archive, destination, digest)['vggt_used'] is False
    assert (destination / 'images/19_001.jpg').read_bytes() == b'fixture photograph'
    with pytest.raises(FileExistsError):
        prepared.extract_prepared(archive, destination, digest)


@pytest.mark.parametrize('changes,message', [
    ({'images/19_001.jpg': b'tampered'}, 'Photograph differs'),
    ({'segment-scene/images/19_001.jpg': b'other photograph'}, 'Segmentation photograph differs'),
    ({'../escaped.txt': b'outside'}, 'Archive path escaped'),
])
def test_bad_inputs_never_extract(prepared, tmp_path, changes, message):
    archive, digest = input_archive(tmp_path, **changes)
    with pytest.raises(ValueError, match=message):
        prepared.extract_prepared(archive, tmp_path / 'scene', digest)
    assert not (tmp_path / 'scene').exists()
    assert not (tmp_path / 'escaped.txt').exists()


def test_recovered_zip_keeps_masks_camera_frame_and_exact_final_ply(prepared, tmp_path):
    run = prepared.PreparedRun(tmp_path, tmp_path / 'project', {'train_steps': 7000})
    run.world.mkdir()
    (run.world / 'world.json').write_text('{}')
    run.execute([sys.executable, '-c', 'pass'], 'pipeline')
    paths = {'segment-scene/sparse/images.bin': b'camera coordinates',
             'segment-scene/masks/19_001.npy': b'object labels',
             'segment-scene/masks/labels.json': b'{"1":"chair"}',
             'gs/ply/point_cloud_6999.ply': b'raw final splats'}
    for relative, content in paths.items():
        target = run.scene / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    with zipfile.ZipFile(run.bundle()) as archive:
        assert archive.testzip() is None
        for relative, content in paths.items():
            assert archive.read('scene/' + relative) == content
    (run.scene / 'gs/ply/point_cloud_6999.ply').rename(run.scene / 'gs/ply/point_cloud_4665.ply')
    with pytest.raises(FileNotFoundError, match='final raw PLY'):
        run.bundle()
