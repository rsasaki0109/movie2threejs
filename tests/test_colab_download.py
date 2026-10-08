import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('colab_download', Path(__file__).parents[1] / 'scripts/colab_download.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)
pack_spec = importlib.util.spec_from_file_location('colab_pack', Path(__file__).parents[1] / 'scripts/colab_pack.py')
packing = importlib.util.module_from_spec(pack_spec)
pack_spec.loader.exec_module(packing)


def fixture():
    data = b'Gaussian weights fixture\x00' * 200
    chunks = [data[:1500], data[1500:]]
    name = 'ckpt_6999_rank0.pt'
    manifest = {'checkpoint_name': name, 'step': 6999, 'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest(), 'parts': [
                    {'name': f'{name}.part{i:03d}', 'bytes': len(c), 'sha256': hashlib.sha256(c).hexdigest()}
                    for i, c in enumerate(chunks)]}
    return manifest, chunks, data


def test_parallel_collection_retries_existing_corrupt_chunk_and_reuses_valid_data(tmp_path):
    manifest, chunks, data = fixture()
    (tmp_path / manifest['parts'][0]['name']).write_bytes(chunks[0])
    (tmp_path / manifest['parts'][1]['name']).write_bytes(b'corrupt prior attempt')
    calls = []
    def fetch(name, target, size):
        calls.append(name)
        target.write_bytes(chunks[int(name[-3:])])
    output = download.collect(manifest, tmp_path, fetch, expected_step=6999)
    assert output.read_bytes() == data
    assert calls == [manifest['parts'][1]['name']]
    download.collect(manifest, tmp_path, lambda *a: pytest.fail('Already verified output downloaded again'))


def test_corrupt_response_never_becomes_a_complete_checkpoint(tmp_path):
    manifest, _, _ = fixture()
    def fetch(name, target, size):
        target.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        download.collect(manifest, tmp_path, fetch)
    assert not (tmp_path / manifest['checkpoint_name']).exists()


def test_pack_and_parallel_download_preserve_real_binary_bytes(tmp_path):
    source = tmp_path / 'ckpt_6999_rank0.pt'
    data = bytes(range(256)) * 500
    source.write_bytes(data)
    parts = tmp_path / 'remote'
    manifest = packing.pack(source, parts, chunk_bytes=8192, step=6999)
    def fetch(name, destination, expected_bytes):
        destination.write_bytes((parts / name).read_bytes())
    result = download.collect(manifest, tmp_path / 'local', fetch, workers=8, expected_step=6999)
    assert result.read_bytes() == data
    with pytest.raises(FileExistsError):
        packing.pack(source, parts)


@pytest.mark.parametrize('change', ['step', 'name', 'size'])
def test_rejects_wrong_final_step_unsafe_path_and_inconsistent_sizes(tmp_path, change):
    manifest, _, _ = fixture()
    if change == 'step':
        manifest['step'] = 4665
    elif change == 'name':
        manifest['parts'][0]['name'] = '../escape'
    else:
        manifest['parts'][0]['bytes'] += 1
    with pytest.raises(ValueError):
        download.collect(manifest, tmp_path, lambda *a: pytest.fail('Invalid manifest fetched data'), expected_step=6999)
