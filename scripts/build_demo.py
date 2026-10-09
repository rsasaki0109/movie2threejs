"""Package the measured public world for static hosting; no GPU or retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from playworld import splat_io

ROOT = Path(__file__).resolve().parents[1]


def pack_splat(source: Path, target: Path) -> int:
    """Write the public 32-byte .splat layout, retaining every Gaussian.

    Layout: xyz float32, exp(scale) float32, DC RGB/opacity uint8,
    normalized wxyz quaternion uint8. Higher SH coefficients are omitted.
    Reference format: https://github.com/antimatter15/splat/blob/main/convert.py
    """
    props = splat_io.read_ply(source)
    n = len(props["x"])
    position = splat_io.means(props).astype("<f4")
    scale = np.exp(np.stack([props[f"scale_{i}"] for i in range(3)], axis=1)).astype("<f4")
    rot = np.stack([props[f"rot_{i}"] for i in range(4)], axis=1)
    rot /= np.maximum(np.linalg.norm(rot, axis=1, keepdims=True), 1e-12)
    color = np.stack([.5 + splat_io.SH_C0 * props[f"f_dc_{i}"] for i in range(3)], axis=1)
    rgba = np.column_stack([color, splat_io.opacity(props)])
    if not all(np.isfinite(v).all() for v in [position, scale, rot, rgba]):
        raise ValueError(f"non-finite Gaussian in {source}")
    rows = np.empty((n, 32), dtype=np.uint8)
    rows[:, :12] = np.ascontiguousarray(position).view(np.uint8).reshape(n, 12)
    rows[:, 12:24] = np.ascontiguousarray(scale).view(np.uint8).reshape(n, 12)
    rows[:, 24:28] = np.clip(rgba * 255, 0, 255).astype(np.uint8)
    rows[:, 28:32] = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(rows.tobytes())
    return n


def pack_spz(source: Path, target: Path) -> int:
    """Preserve SH3 in SPZ v3. Requires the optional Niantic SPZ bindings.

    The capture coordinates are object-local coordinates: world.align already
    maps them to Three.js. Apply no extra RDF/RUB flip here. Spark's SPZ reader
    uses the stored axes directly, as verified against the PLY/.splat viewer.
    """
    import spz
    cloud = spz.load_splat_from_ply(str(source))
    if cloud.num_points < 1:
        raise ValueError(f"empty Gaussian cloud: {source}")
    options = spz.PackOptions()
    options.from_coord = spz.CoordinateSystem.RUB
    options.version = 3
    target.parent.mkdir(parents=True, exist_ok=True)
    if not spz.save_spz(cloud, options, str(target)):
        raise RuntimeError(f"SPZ export failed: {target}")
    return cloud.num_points


def package_world(source: Path, out: Path, profile: dict | None = None, format: str = "splat"):
    if format not in {"splat", "spz"}:
        raise ValueError(f"unsupported splat format: {format}")
    out.mkdir(parents=True, exist_ok=True)
    world = json.loads((source / "world.json").read_text())
    paths = [world["background"], *(o["splat"] for o in world["objects"])]
    files = []
    for path in paths:
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe world path {path}")
        packed = relative.with_suffix("." + format)
        count = (pack_spz if format == "spz" else pack_splat)(source / relative, out / packed)
        files.append({"path": packed.as_posix(), "gaussians": count,
                      "bytes": (out / packed).stat().st_size,
                      "sha256": hashlib.sha256((out / packed).read_bytes()).hexdigest()})
    world["background"] = Path(world["background"]).with_suffix("." + format).as_posix()
    for obj in world["objects"]:
        obj["splat"] = Path(obj["splat"]).with_suffix("." + format).as_posix()
    photo_files = []
    for backdrop in world.get('photo_backdrops', []):
        relative = Path(backdrop['texture'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'unsafe photo backdrop path {relative}')
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
        photo_files.append({'path':relative.as_posix(), 'bytes':target.stat().st_size,
                            'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    if profile is not None:
        world["player"].update(profile.get("player", {}))
        world["demo"] = profile.get("demo", {})
    (out / "world.json").write_text(json.dumps(world, separators=(",", ":")), encoding="utf-8")
    manifest = {"source_world_sha256": hashlib.sha256((source / "world.json").read_bytes()).hexdigest(),
                "format": "Niantic SPZ v3" if format == "spz" else "antimatter15 .splat (DC color, 8-bit opacity/rotation)",
                "higher_order_sh": "preserved through SH3 (quantized)" if format == "spz" else "omitted for download size", "files": files,
                "gaussians": sum(f["gaussians"] for f in files),
                "splat_bytes": sum(f["bytes"] for f in files),
                "objects": len(world["objects"]), "colliders": len(world["colliders"]),
                "photo_backdrops": photo_files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


def build_site(assets: Path, out: Path):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    shutil.copytree(assets, out, dirs_exist_ok=True)
    shutil.copytree(ROOT / "web/demo", out, dirs_exist_ok=True)
    for name in ["main.js", "record.js"]:
        shutil.copy2(ROOT / "src/playworld/viewer" / name, out / name)
    for name in ["EYEFULTOWER-LICENSE.txt", "data-attribution.md", "hero.mp4", "hero.gif"]:
        specific = assets / name
        shutil.copy2(specific if specific.exists() else ROOT / "docs" / name, out / name)
    if not (out / 'hero-shot.json').exists():
        shutil.copy2(ROOT / "shots/hero.json", out / "hero-shot.json")
    (out / ".nojekyll").touch()
    print(f"Static site: {out} ({sum(p.stat().st_size for p in out.rglob('*') if p.is_file()):,} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, help="Convert a world into --assets once")
    parser.add_argument("--assets", type=Path, default=ROOT / "demo-assets/workbench")
    parser.add_argument("--profile", type=Path, help="Optional player and featured action overrides")
    parser.add_argument("--format", choices=["splat", "spz"], default="splat", help="SPZ retains view-dependent color; install optional Niantic SPZ bindings")
    parser.add_argument("--out", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    if args.world:
        profile = json.loads(args.profile.read_text()) if args.profile else None
        package_world(args.world, args.assets, profile, args.format)
    build_site(args.assets, args.out)


if __name__ == "__main__":
    main()
