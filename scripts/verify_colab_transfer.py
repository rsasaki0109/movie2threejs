"""Validate recovery of Colab-sized synthetic files over local HTTP, without a GPU.

This does not validate the Colab proxy, actual model weights or reconstruction.
Writes several GB under a fresh --workdir and retains them for inspection.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import threading
import time
import urllib.request
import zipfile

from colab_download import download_index, sha256
from notebook_support import NotebookRun


def write_payload(path, size, block):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as destination:
        remaining = size
        while remaining:
            count = min(len(block), remaining)
            destination.write(block[:count])
            remaining -= count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint-bytes', type=int, default=441186741)
    parser.add_argument('--world-bytes', type=int, default=405692278)
    parser.add_argument('--prepare-only', action='store_true',
                        help='Prepare synthetic chunks on Colab, without a local HTTP server or download')
    args = parser.parse_args()
    if args.checkpoint_bytes <= 8 * 1024 * 1024 or args.world_bytes <= 0:
        parser.error('Checkpoint fixture must exceed 8 MiB; world fixture must be positive')
    work = args.workdir.resolve()
    work.parent.mkdir(parents=True, exist_ok=True)
    required = 5 * (args.checkpoint_bytes + args.world_bytes) + 64 * 1024 * 1024
    if shutil.disk_usage(work.parent).free < required:
        raise RuntimeError('Insufficient disk space for source, chunks, ZIP and recovered copies')
    work.mkdir(exist_ok=False)
    content = work / 'content'
    run = NotebookRun(content, work / 'project', {'train_steps': 7000, 'synthetic': True})
    started = time.monotonic()
    block = os.urandom(8 * 1024 * 1024)
    checkpoint = run.scene / 'gs' / 'ckpts' / 'ckpt_6999_rank0.pt'
    write_payload(checkpoint, args.checkpoint_bytes, block)
    write_payload(run.world / 'synthetic.bin', args.world_bytes, block)
    (run.world / 'world.json').write_text('{"synthetic_transfer_fixture": true}\n', encoding='utf-8')
    run.report['completed'] = True  # Synthetic fixture; no inference or model load.
    run.save()
    preparation_start = time.monotonic()
    index_path = run.prepare_backups()
    preparation_seconds = time.monotonic() - preparation_start
    index = json.loads(index_path.read_text(encoding='utf-8'))
    if args.prepare_only:
        report = {'format': 'playworld-transfer-fixture/1', 'synthetic_files': True,
                  'gpu_used': False, 'download_verified': False,
                  'backup_index': index_path.as_posix(),
                  'preparation_seconds': round(preparation_seconds, 3)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))
        return
    remote_index = '/content/' + index_path.relative_to(content).as_posix()
    for artifact in index['artifacts']:
        artifact['directory'] = '/content/' + Path(artifact['directory']).relative_to(content).as_posix()
    index_path.write_text(json.dumps(index), encoding='utf-8')
    requests = {}
    lock = threading.Lock()
    injected = []

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            # One intentionally truncated HTTP response exercises the real retry.
            with lock:
                requests[self.path] = requests.get(self.path, 0) + 1
                truncate = self.path.endswith('ckpt_6999_rank0.pt.part001') and requests[self.path] == 1
            if truncate:
                source = content / PurePosixPath(self.path).relative_to('/content')
                body = source.read_bytes()[:1024]
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                injected.append('truncated_checkpoint_chunk')
                return
            super().do_GET()

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(work)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = work / 'recovered'
    # Simulate a prior interrupted attempt with one verified chunk and one bad chunk.
    checkpoint_parts = content / PurePosixPath(index['artifacts'][0]['directory']).relative_to('/content')
    resumed = destination / 'final_checkpoint'
    resumed.mkdir(parents=True)
    shutil.copy(checkpoint_parts / 'ckpt_6999_rank0.pt.part000', resumed)
    (resumed / 'ckpt_6999_rank0.pt.part001').write_bytes(b'interrupted')
    def open_file(name):
        return urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}{name}', timeout=30)
    try:
        transfer_start = time.monotonic()
        results = download_index(open_file, remote_index, destination, 6999, wait_seconds=30)
        transfer_seconds = time.monotonic() - transfer_start
        archive = Path(results['world_zip']['file'])
        with zipfile.ZipFile(archive) as bundle:
            assert bundle.testzip() is None
            assert json.loads(bundle.read('world/world.json'))['synthetic_transfer_fixture']
        assert sha256(Path(results['final_checkpoint']['file'])) == sha256(checkpoint)
        checkpoint_url = index['artifacts'][0]['directory'] + '/ckpt_6999_rank0.pt.part000'
        assert requests.get(checkpoint_url, 0) == 0
        assert injected == ['truncated_checkpoint_chunk']
        report = {
            'format': 'playworld-local-http-transfer/1', 'platform': platform.platform(),
            'python': platform.python_version(), 'transport': 'loopback HTTP, native Python',
            'gpu_used': False, 'synthetic_files': True, 'real_colab_proxy_verified': False,
            'actual_checkpoint_cpu_load_verified': False,
            'preparation_seconds': round(preparation_seconds, 3),
            'transfer_seconds': round(transfer_seconds, 3),
            'total_seconds': round(time.monotonic() - started, 3),
            'artifacts': {kind: {k: v for k, v in result.items() if k != 'file'}
                          for kind, result in results.items()},
            'archive_integrity_verified': True, 'whole_file_hashes_verified': True,
            'valid_chunk_reused': True, 'corrupt_chunk_replaced': True,
            'truncated_http_response_retried': True, 'workers': 8,
            'limits': 'Local synthetic HTTP transfer only; not Colab throughput or a recoverable GPU scene.',
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == '__main__':
    main()
