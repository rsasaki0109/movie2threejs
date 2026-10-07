"""Download a small, MIT-licensed public apartment capture for pipeline validation.

This is a single-camera visualization of sampled photographs from a capture rig,
not a smartphone video. Preserve that distinction in the GIF/README.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

DATA_URL = "https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/EyefulTower/apartment/images-jpeg-2k/19.mp4"
REPO = "https://github.com/facebookresearch/EyefulTower"


def download(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    revision = json.load(urllib.request.urlopen(urllib.request.Request(
        "https://api.github.com/repos/facebookresearch/EyefulTower/commits/main",
        headers={"User-Agent": "playworld-public-capture"}), timeout=30))["sha"]
    with urllib.request.urlopen(f"https://raw.githubusercontent.com/facebookresearch/EyefulTower/{revision}/README.md", timeout=30) as response:
        readme = response.read().decode()
    if "Relicensed all content under the MIT license" not in readme:
        raise RuntimeError("Upstream dataset license statement changed; inspect before use")
    with urllib.request.urlopen(f"https://raw.githubusercontent.com/facebookresearch/EyefulTower/{revision}/LICENSE", timeout=30) as response:
        notice = response.read()
    if not notice.startswith(b"MIT License"):
        raise RuntimeError("Expected MIT license notice")
    (out / "EYEFULTOWER-LICENSE.txt").write_bytes(notice)
    target = out / "apartment-camera19.mp4"
    if not target.exists():
        temporary = target.with_suffix(".mp4.part")
        with urllib.request.urlopen(DATA_URL, timeout=60) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = {"dataset": "Eyeful Tower / apartment / camera 19", "url": DATA_URL,
                "repository": REPO, "license": "MIT", "license_revision": revision,
                "sha256": digest, "kind": "capture-rig photograph sequence visualization",
                "citation": "Xu et al., VR-NeRF: High-Fidelity Virtualized Walkable Spaces, SIGGRAPH Asia 2023. DOI: 10.1145/3610548.3618139",
                "readme_url": f"{REPO}/blob/{revision}/README.md"}
    (out / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return target


def workshop(video: Path) -> Path:
    """The short, overlapping workbench section used for the measured L4 run."""
    target = video.with_name('apartment-workshop.mp4')
    if not target.exists():
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', '5.0',
                        '-i', str(video), '-t', '4.2', '-map', '0:v:0', '-an',
                        '-c:v', 'libx264', '-crf', '15', '-preset', 'fast',
                        '-pix_fmt', 'yuv420p', str(target)], check=True)
    details = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration,size', '-of', 'json', str(target)]))
    metadata = {'source': video.name, 'source_sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
                'derived': target.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'start_seconds': 5.0, 'requested_duration_seconds': 4.2,
                'measured_duration_seconds': float(details['format']['duration']),
                'bytes': target.stat().st_size, 'license': 'MIT',
                'kind': 'trimmed capture-rig photograph sequence visualization'}
    target.with_suffix('.source.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("scenes/public_source"))
    parser.add_argument('--workshop', action='store_true', help='trim the verified overlapping workbench section')
    args = parser.parse_args()
    video = download(args.out)
    if args.workshop:
        print(workshop(video), flush=True)
