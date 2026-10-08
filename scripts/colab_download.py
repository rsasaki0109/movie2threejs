"""Download hashed Colab chunks in parallel without logging proxy credentials.

Requires an already connected session in the official google-colab-cli. It never
allocates or stops a runtime. Real Colab transfers still need validation; the
integrity, resume and path checks are covered by local fixtures.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
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
        list(pool.map(download, manifest['parts']))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--remote-directory', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    parser.add_argument('--sdk-python', default='/root/.local/share/uv/tools/google-colab-cli/bin/python')
    parser.add_argument('--distro', default='Ubuntu-22.04')
    args = parser.parse_args()
    remote = PurePosixPath(args.remote_directory)
    if not remote.is_absolute() or len(remote.parts) < 2 or remote.parts[1] != 'content' or '..' in remote.parts:
        raise ValueError('Expected a directory under /content')
    info = runtime(args.session, args.sdk_python, args.distro)
    query = urllib.parse.urlencode({'authuser': '0', 'colab-runtime-proxy-token': info['token']})

    def open_file(name):
        url = info['url'].rstrip('/') + '/files/' + urllib.parse.quote(str(remote / name).lstrip('/'), safe='/') + '?' + query
        return urllib.request.urlopen(url, timeout=30)

    def fetch(name, target, expected_bytes):
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
                return
            except Exception:
                # Exception strings can include the credential-bearing URL.
                if attempt == 2:
                    raise RuntimeError(f'File transfer failed: {name}') from None
    try:
        with open_file('manifest.json') as response:
            data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError('Manifest is too large')
        manifest = json.loads(data)
    except Exception:
        raise RuntimeError('Cannot read the manifest from the existing runtime') from None
    output = collect(manifest, args.out, fetch, args.workers, args.expected_step)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'file': str(output), 'bytes': output.stat().st_size, 'sha256': sha256(output)}))


if __name__ == '__main__':
    main()
