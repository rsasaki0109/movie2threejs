"""Bundle repository sources for Colab, excluding environments and private captures."""
import argparse
import subprocess
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=root / ".cache/playworld-colab.zip")
parser.add_argument('--sources-only', action='store_true', help='omit rendered assets and website files from the GPU job upload')
args = parser.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)
paths = subprocess.check_output(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
).decode().split("\0")
with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(set(paths) - {""}):
        if args.sources_only and Path(name).parts[0] in {'demo-assets', 'docs', 'web', 'shots', '.github'}:
            continue
        path = root / name
        if path.is_file() and path.resolve() != args.out.resolve():
            archive.write(path, arcname="playworld/" + name)
print(f"{args.out} ({args.out.stat().st_size:,} bytes)")
