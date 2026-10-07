"""playworld: phone video -> walkable, physical three.js world.

Stages (each can be run on its own):
  frames  video -> scene/images/*.png
  poses   scene/images -> scene/sparse (VGGT feed-forward, COLMAP format)
  train   scene -> scene/gs/ply/*.ply (gsplat)
  segment scene/images -> scene/masks (SAM 3, movable things by text prompt)
  world   sparse + splats (+ masks) -> out/ (world.json, splats, viewer)
  all     everything above
  demo    synthetic room, no GPU needed
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def extract_frames(video: Path, scene: Path, num: int, max_size: int) -> None:
    out = scene / "images"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    duration = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    )
    fps = num / max(duration, 1e-3)
    scale = f"scale='if(gt(iw,ih),min({max_size},iw),-2)':'if(gt(iw,ih),-2,min({max_size},ih))'"
    run(["ffmpeg", "-loglevel", "error", "-i", video, "-vf", f"fps={fps:.6f},{scale}", "-frames:v", str(num), out / "frame_%04d.png"])
    print(f"extracted {len(list(out.glob('*.png')))} frames")


def estimate_poses(scene: Path, vggt_dir: Path, use_ba: bool, python: str = sys.executable) -> None:
    cmd = [python, Path(vggt_dir) / "demo_colmap.py", f"--scene_dir={Path(scene).resolve()}"]
    if use_ba:
        cmd.append("--use_ba")
    run(cmd, cwd=vggt_dir)


def train_splats(scene: Path, gsplat_dir: Path, steps: int, python: str = sys.executable) -> Path:
    scene = Path(scene).resolve()
    run(
        [
            python, Path(gsplat_dir) / "examples" / "simple_trainer.py", "default",
            "--data-dir", scene, "--data-factor", "1", "--result-dir", scene / "gs",
            "--max-steps", str(steps), "--save-ply", "--ply-steps", str(steps),
            "--eval-steps", str(steps), "--save-steps", str(steps),
            "--no-normalize-world-space",  # keep the COLMAP frame so splats and cameras agree
            "--test-every", "100000", "--disable-viewer",
        ],
        cwd=Path(gsplat_dir) / "examples",
    )
    return latest_ply(scene)


def segment_objects(scene: Path, prompts: str, python: str = sys.executable) -> None:
    # The SAM 3 environment does not need playworld installed: run the module file directly.
    run([python, Path(__file__).with_name("segment.py"), Path(scene).resolve(), "--prompts", prompts])


def latest_ply(scene: Path) -> Path:
    plys = sorted((scene / "gs" / "ply").glob("*.ply"), key=lambda p: p.stat().st_mtime)
    if not plys:
        raise FileNotFoundError(f"no splat ply under {scene / 'gs' / 'ply'}")
    return plys[-1]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="playworld", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("frames")
    p.add_argument("video", type=Path)
    p.add_argument("scene", type=Path)
    p.add_argument("--num", type=int, default=48, help="frames to keep (VGGT memory grows with this)")
    p.add_argument("--max-size", type=int, default=1280)

    p = sub.add_parser("poses")
    p.add_argument("scene", type=Path)
    p.add_argument("--vggt-dir", type=Path, required=True)
    p.add_argument("--ba", action="store_true", help="bundle adjustment (slower, sharper)")
    p.add_argument("--python", default=sys.executable, help="interpreter of the VGGT environment")

    p = sub.add_parser("train")
    p.add_argument("scene", type=Path)
    p.add_argument("--gsplat-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=7000)
    p.add_argument("--python", default=sys.executable, help="interpreter of the gsplat environment")

    from .segment import DEFAULT_PROMPTS

    p = sub.add_parser("segment")
    p.add_argument("scene", type=Path)
    p.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS))
    p.add_argument("--python", default=sys.executable, help="interpreter of the SAM 3 environment")

    def world_args(p):
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--masks", type=Path, help="per-frame instance masks (<stem>.npy + labels.json)")
        p.add_argument("--eye-height", type=float, default=1.5, help="phone height above floor in meters")
        p.add_argument("--voxel", type=float, default=0.1)

    p = sub.add_parser("world")
    p.add_argument("scene", type=Path)
    p.add_argument("--splats", type=Path, help="defaults to the latest gsplat export")
    world_args(p)

    p = sub.add_parser("all")
    p.add_argument("video", type=Path)
    p.add_argument("scene", type=Path)
    p.add_argument("--vggt-dir", type=Path, required=True)
    p.add_argument("--gsplat-dir", type=Path, required=True)
    p.add_argument("--num", type=int, default=48)
    p.add_argument("--max-size", type=int, default=1280)
    p.add_argument("--ba", action="store_true")
    p.add_argument("--steps", type=int, default=7000)
    p.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS), help="movable things; empty = static world only")
    p.add_argument("--vggt-python", default=sys.executable)
    p.add_argument("--gsplat-python", default=sys.executable)
    p.add_argument("--sam3-python", default=sys.executable)
    world_args(p)

    p = sub.add_parser("demo")
    p.add_argument("--out", type=Path, default=Path("demo_world"))
    p.add_argument("--seed", type=int, default=0)

    a = ap.parse_args(argv)
    from .world import build_world  # numpy/scipy import deferred so --help is instant

    timings = {}

    def stage(name, fn, *args):
        t0 = time.time()
        r = fn(*args)
        timings[name] = round(time.time() - t0, 1)
        print(f"[{name}] {timings[name]} s", flush=True)
        return r

    if a.cmd == "frames":
        extract_frames(a.video, a.scene, a.num, a.max_size)
    elif a.cmd == "poses":
        estimate_poses(a.scene, a.vggt_dir, a.ba, a.python)
    elif a.cmd == "train":
        print(train_splats(a.scene, a.gsplat_dir, a.steps, a.python))
    elif a.cmd == "segment":
        segment_objects(a.scene, a.prompts, a.python)
    elif a.cmd == "world":
        masks = a.masks or ((a.scene / "masks") if (a.scene / "masks" / "labels.json").exists() else None)
        w = build_world(a.scene / "sparse", a.splats or latest_ply(a.scene), a.out, masks, a.eye_height, a.voxel)
        print(json.dumps(w["stats"], indent=1))
    elif a.cmd == "all":
        stage("frames", extract_frames, a.video, a.scene, a.num, a.max_size)
        stage("poses", estimate_poses, a.scene, a.vggt_dir, a.ba, a.vggt_python)
        ply = stage("train", train_splats, a.scene, a.gsplat_dir, a.steps, a.gsplat_python)
        masks = a.masks
        if masks is None and a.prompts.strip():
            stage("segment", segment_objects, a.scene, a.prompts, a.sam3_python)
            masks = a.scene / "masks"
        w = stage("world", build_world, a.scene / "sparse", ply, a.out, masks, a.eye_height, a.voxel)
        print(json.dumps({**w["stats"], "seconds": timings}, indent=1))
    elif a.cmd == "demo":
        from .synthetic import make_scene

        make_scene(a.out / "_capture", seed=a.seed)
        w = build_world(a.out / "_capture" / "sparse", a.out / "_capture" / "splats.ply", a.out, a.out / "_capture" / "masks")
        print(json.dumps(w["stats"], indent=1))
        print(f"serve with: python -m http.server -d {a.out} 8000  ->  http://localhost:8000/")


if __name__ == "__main__":
    main()
