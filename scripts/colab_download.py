"""Download hashed Colab chunks in parallel without logging proxy credentials.

Requires an already connected session in the official google-colab-cli. It never
allocates or stops a runtime. Integrity, resume and path checks are covered by
local fixtures. A subsequent L4 run recovered both final GPU artifacts before
termination with a 32-worker index watcher and a concurrent ZIP transfer.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest, expected_step=None):
    name = manifest.get('artifact_name', manifest.get('checkpoint_name'))
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
        raise ValueError('Unsafe artifact name')
    if expected_step is not None and manifest.get('step') != expected_step:
        raise ValueError('Not the expected final checkpoint step')
    if expected_step is not None and name != f'ckpt_{expected_step}_rank0.pt':
        raise ValueError('Not the expected final checkpoint filename')
    parts = manifest.get('parts')
    if not isinstance(parts, list) or not parts or len(parts) > 1000:
        raise ValueError('Invalid chunk list')
    for index, part in enumerate(parts):
        if part.get('name') != f'{name}.part{index:03d}' or not isinstance(part.get('bytes'), int) or part['bytes'] <= 0:
            raise ValueError('Unsafe or out-of-order chunk')
        if not re.fullmatch(r'[0-9a-f]{64}', part.get('sha256', '')):
            raise ValueError('Invalid chunk digest')
    if sum(p['bytes'] for p in parts) != manifest.get('bytes'):
        raise ValueError('Chunk sizes do not match the artifact')
    if not re.fullmatch(r'[0-9a-f]{64}', manifest.get('sha256', '')):
        raise ValueError('Invalid artifact digest')
    return name


def collect(manifest, directory, fetch, workers=8, expected_step=None):
    name = validate_manifest(manifest, expected_step)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / name
    if output.exists():
        if output.stat().st_size != manifest['bytes'] or sha256(output) != manifest['sha256']:
            raise ValueError('Existing output differs; choose a fresh directory')
        return output

    def download(part):
        target = directory / part['name']
        if not target.resolve().is_relative_to(directory):
            raise ValueError('Chunk escapes the output directory')
        if target.exists() and target.stat().st_size == part['bytes'] and sha256(target) == part['sha256']:
            return
        temporary = target.with_suffix(target.suffix + '.partial')
        fetch(part['name'], temporary, part['bytes'])
        if temporary.stat().st_size != part['bytes'] or sha256(temporary) != part['sha256']:
            raise ValueError('Downloaded chunk failed integrity checks')
        temporary.replace(target)
        print(f"Verified {part['name']} ({part['bytes']} bytes)", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download, part) for part in manifest['parts']]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    temporary = output.with_suffix(output.suffix + '.partial')
    with temporary.open('wb') as destination:
        for part in manifest['parts']:
            with (directory / part['name']).open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    destination.write(block)
    if temporary.stat().st_size != manifest['bytes'] or sha256(temporary) != manifest['sha256']:
        raise ValueError('Assembled artifact failed integrity checks')
    temporary.replace(output)
    return output


def runtime(session, sdk_python, distro):
    program = "from colab_cli.common import state; import json; s=state.get_session(" + repr(session) + "); print(json.dumps({'url':s.url,'token':s.token}))"
    command = [sdk_python, '-c', program]
    if sys.platform == 'win32':
        command = ['wsl', '-d', distro, '--exec', *command]
    # Credentials travel only through a captured pipe into this process's memory.
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('Cannot read the existing Colab session; no runtime was allocated')
    info = json.loads(result.stdout)
    parsed = urllib.parse.urlparse(info['url'])
    if parsed.scheme != 'https' or not parsed.hostname or not parsed.hostname.endswith('.prod.colab.dev'):
        raise ValueError('Unexpected Colab proxy host')
    return info


def read_manifest(open_file, name='manifest.json'):
    try:
        with open_file(name) as response:
            data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError('Manifest is too large')
        return json.loads(data)
    except Exception:
        raise RuntimeError('Cannot read the manifest from the existing runtime') from None


def fetch_file(open_file, name, target, expected_bytes, expected_sha256=None):
    """Retry truncated responses and corrupt bytes; never expose request URLs."""
    for attempt in range(3):
        try:
            with open_file(name) as response, target.open('wb') as destination:
                total = 0
                for block in iter(lambda: response.read(256 * 1024), b''):
                    total += len(block)
                    if total > expected_bytes:
                        raise ValueError('Response is larger than the expected chunk')
                    destination.write(block)
                if total != expected_bytes:
                    raise ValueError('Response ended before the expected chunk size')
            if expected_sha256 is not None and sha256(target) != expected_sha256:
                raise ValueError('Response digest differs from the expected chunk')
            return
        except Exception:
            # Exception strings can include the credential-bearing URL.
            if attempt == 2:
                raise RuntimeError(f'File transfer failed: {name}') from None


def download_manifest(open_file, manifest, directory, workers=8, expected_step=None):
    validate_manifest(manifest, expected_step)
    hashes = {part['name']: part['sha256'] for part in manifest['parts']}
    def fetch(name, target, size):
        fetch_file(open_file, name, target, size, hashes[name])
    output = collect(manifest, directory, fetch, workers, expected_step)
    (Path(directory) / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return output


def content_path(value):
    remote = PurePosixPath(value)
    if not remote.is_absolute() or len(remote.parts) < 2 or remote.parts[1] != 'content' or '..' in remote.parts:
        raise ValueError('Expected a path under /content')
    return remote


def download_index(open_file, remote_index, directory, expected_step, workers=8,
                   wait_seconds=1800, poll_seconds=5):
    """Recover each ready artifact while the notebook prepares the next one."""
    remote_index = content_path(remote_index)
    directory = Path(directory)
    deadline = time.monotonic() + wait_seconds
    verified = {}
    while True:
        try:
            index = read_manifest(open_file, str(remote_index))
        except RuntimeError:
            index = None  # The notebook may still be reconstructing.
        if index is not None:
            if index.get('format') != 'playworld-backup-index/1':
                raise ValueError('Unexpected backup index format')
            for artifact in index.get('artifacts', []):
                kind = artifact.get('kind')
                if kind not in ('final_checkpoint', 'world_zip') or kind in verified:
                    continue
                remote = content_path(artifact['directory'])
                if not remote.is_relative_to(remote_index.parent):
                    raise ValueError('Artifact escapes the attempt directory')
                def artifact_file(name):
                    return open_file(str(remote / name))
                try:
                    manifest = read_manifest(artifact_file)
                    output = download_manifest(artifact_file, manifest, directory / kind, workers,
                                               expected_step if kind == 'final_checkpoint' else None)
                except RuntimeError:
                    # A flaky connection must not discard the whole watch attempt.
                    # collect() preserves verified chunks; the next poll reuses them.
                    print(f'Transfer interrupted: {kind}; retrying verified chunks', flush=True)
                    break
                verified[kind] = {'file': str(output.resolve()), 'bytes': output.stat().st_size,
                                  'sha256': sha256(output)}
                (directory / 'download-receipt.json').write_text(
                    json.dumps({'format': 'playworld-backup-download/1', 'artifacts': verified,
                                'complete': len(verified) == 2}, indent=2) + '\n', encoding='utf-8')
            if index.get('complete') and len(verified) == 2:
                return verified
            if index.get('error_type'):
                raise RuntimeError('Remote backup preparation failed; any verified artifact was preserved')
        if time.monotonic() >= deadline:
            raise TimeoutError('Backup index wait expired; any verified artifact was preserved')
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--remote-directory')
    source.add_argument('--remote-index', help='Watch the notebook backup index and recover both artifacts')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 33))
    parser.add_argument('--wait-seconds', type=int, default=1800, help='Maximum index wait; does not allocate GPU time')
    parser.add_argument('--sdk-python', default='/root/.local/share/uv/tools/google-colab-cli/bin/python')
    parser.add_argument('--distro', default='Ubuntu-22.04')
    args = parser.parse_args()
    remote = content_path(args.remote_directory) if args.remote_directory else None
    if args.remote_index and args.expected_step is None:
        parser.error('--remote-index requires --expected-step')
    if args.remote_index:
        content_path(args.remote_index)
    info = runtime(args.session, args.sdk_python, args.distro)
    query = urllib.parse.urlencode({'authuser': '0', 'colab-runtime-proxy-token': info['token']})

    def open_file(name):
        path = remote / name if remote is not None else content_path(name)
        url = info['url'].rstrip('/') + '/files/' + urllib.parse.quote(str(path).lstrip('/'), safe='/') + '?' + query
        return urllib.request.urlopen(url, timeout=30)

    if args.remote_index:
        print(json.dumps(download_index(open_file, args.remote_index, args.out,
                                        args.expected_step, args.workers, args.wait_seconds)))
    else:
        output = download_manifest(open_file, read_manifest(open_file), args.out, args.workers, args.expected_step)
        print(json.dumps({'file': str(output), 'bytes': output.stat().st_size, 'sha256': sha256(output)}))


if __name__ == '__main__':
    main()
