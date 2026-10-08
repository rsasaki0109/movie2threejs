"""Prepare hashed download chunks on an existing runtime; no GPU is required."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def pack(source, directory, chunk_bytes=8 * 1024 * 1024, step=None):
    source, directory = Path(source), Path(directory)
    if not source.is_file() or source.stat().st_size == 0 or chunk_bytes <= 0:
        raise ValueError('Expected a nonempty artifact and positive chunk size')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', source.name):
        raise ValueError('Use an ASCII artifact filename without path separators')
    if step is not None and source.name != f'ckpt_{step}_rank0.pt':
        raise ValueError('Checkpoint filename differs from the specified step')
    directory.mkdir(parents=True, exist_ok=False)
    whole = hashlib.sha256()
    parts = []
    with source.open('rb') as stream:
        for index in range(1000):
            data = stream.read(chunk_bytes)
            if not data:
                break
            name = f'{source.name}.part{index:03d}'
            (directory / name).write_bytes(data)
            whole.update(data)
            parts.append({'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        if stream.read(1):
            raise ValueError('Artifact requires more than 1000 chunks')
    manifest = {'artifact_name': source.name, 'bytes': sum(p['bytes'] for p in parts),
                'sha256': whole.hexdigest(), 'parts': parts}
    if manifest['bytes'] != source.stat().st_size:
        raise ValueError('Artifact changed during packing')
    if step is not None:
        manifest['step'] = step
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--chunk-mib', type=int, default=8, choices=range(1, 33))
    parser.add_argument('--step', type=int)
    args = parser.parse_args()
    result = pack(args.source, args.directory, args.chunk_mib * 1024 * 1024, args.step)
    print(json.dumps({'manifest': str(args.directory / 'manifest.json'), 'bytes': result['bytes'], 'parts': len(result['parts'])}))
