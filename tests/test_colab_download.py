import hashlib
import importlib.util
import io
import json
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import threading
import urllib.request

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


def test_real_http_retries_truncated_and_same_size_corrupt_response(tmp_path):
    manifest, chunks, data = fixture()
    calls = {}
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            name = self.path.lstrip('/')
            calls[name] = calls.get(name, 0) + 1
            if name == 'manifest.json':
                body = json.dumps(manifest).encode()
            else:
                body = chunks[int(name[-3:])]
                if name.endswith('000') and calls[name] == 1:
                    body = body[:len(body) // 2]
                elif name.endswith('000') and calls[name] == 2:
                    body = bytes([body[0] ^ 1]) + body[1:]
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def open_file(name):
            return urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/{name}', timeout=5)
        output = download.download_manifest(open_file, download.read_manifest(open_file), tmp_path, expected_step=6999)
        assert output.read_bytes() == data
        assert calls[manifest['parts'][0]['name']] == 3
        assert calls[manifest['parts'][1]['name']] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_transfer_errors_exclude_credential_bearing_urls(tmp_path):
    credential = 'fixture-only-secret'
    calls = []
    def fail(name):
        calls.append(name)
        raise OSError(f'https://proxy.example/files?token={credential}')
    with pytest.raises(RuntimeError) as error:
        download.fetch_file(fail, 'weights.part000', tmp_path / 'partial', 12)
    assert credential not in str(error.value)
    assert len(calls) == 3
    with pytest.raises(RuntimeError) as error:
        download.read_manifest(fail)
    assert credential not in str(error.value)


def test_index_recovers_checkpoint_before_world_is_ready_and_keeps_receipt(tmp_path):
    manifest, chunks, data = fixture()
    world = dict(manifest, artifact_name='playworld-result.zip')
    world['parts'] = [dict(p, name=f"playworld-result.zip.part{i:03d}") for i, p in enumerate(manifest['parts'])]
    calls = []
    def open_file(name):
        calls.append(name)
        if name == '/content/run/backup-index.json':
            index_calls = calls.count(name)
            if index_calls == 1:
                raise FileNotFoundError('Not prepared yet')
            artifacts = [{'kind': 'final_checkpoint', 'directory': '/content/run/backup/checkpoint'}]
            if index_calls > 2:
                assert (tmp_path / 'final_checkpoint' / manifest['checkpoint_name']).read_bytes() == data
                artifacts += [{'kind': 'world_zip', 'directory': '/content/run/backup/world'}]
            return io.BytesIO(json.dumps({'format': 'playworld-backup-index/1',
                                         'complete': index_calls > 2, 'artifacts': artifacts}).encode())
        if name.endswith('manifest.json'):
            return io.BytesIO(json.dumps(world if '/world/' in name else manifest).encode())
        return io.BytesIO(chunks[int(name[-3:])])
    results = download.download_index(open_file, '/content/run/backup-index.json', tmp_path, 6999, poll_seconds=0)
    assert set(results) == {'final_checkpoint', 'world_zip'}
    assert json.loads((tmp_path / 'download-receipt.json').read_text())['complete']
    assert calls.count('/content/run/backup/checkpoint/manifest.json') == 1


def test_index_rejects_artifact_outside_the_attempt(tmp_path):
    index = {'format': 'playworld-backup-index/1', 'complete': False,
             'artifacts': [{'kind': 'world_zip', 'directory': '/content/other/world'}]}
    with pytest.raises(ValueError, match='escapes'):
        download.download_index(lambda name: io.BytesIO(json.dumps(index).encode()),
                                '/content/run/backup-index.json', tmp_path, 6999)
