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
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def extract_frames(video: Path, scene: Path, num: int, max_size: int) -> None:
    if not video.is_file():
        raise FileNotFoundError(video)
    if num < 2 or max_size < 32:
        raise ValueError("at least two frames and max-size >= 32 required")
    out = scene / "images"
    if out.exists():
        if not out.resolve().is_relative_to(scene.resolve()):
            raise ValueError("refusing to remove an images directory outside the requested scene")
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


def estimate_poses(scene: Path, vggt_dir: Path, use_ba: bool, python: str = sys.executable,
                   confidence: float = 5.0, shared_camera: bool = False, track_budget: int = 32768,
                   reprojection_error: float = 8.0) -> None:
    import math
    if not math.isfinite(confidence) or confidence <= 0:
        raise ValueError("pose confidence must be a finite positive value")
    vggt_dir = vggt_dir.resolve()
    cmd = [python, Path(__file__).with_name("vggt_runner.py"), "--vggt-dir", vggt_dir,
           "--track-budget", str(track_budget), f"--scene_dir={Path(scene).resolve()}",
           f"--conf_thres_value={confidence}"]
    if use_ba:
        cmd.extend(["--use_ba", "--max_reproj_error", str(reprojection_error)])
    if shared_camera:
        cmd.append("--shared_camera")
    run(cmd, cwd=vggt_dir)
    from .colmap_io import read_model
    import numpy as np
    reconstruction = read_model(Path(scene) / "sparse")
    if len(reconstruction.xyz) < 4 or not np.isfinite(reconstruction.xyz).all():
        raise ValueError("VGGT exported fewer than four finite points; inspect reconstruction/confidence before training (adjust --pose-confidence explicitly if appropriate)")
    print(f"validated {len(reconstruction.images)} poses and {len(reconstruction.xyz)} finite points", flush=True)


def train_splats(scene: Path, gsplat_dir: Path, steps: int, python: str = sys.executable) -> Path:
    scene = Path(scene).resolve()
    gsplat_dir = gsplat_dir.resolve()
    if steps < 1:
        raise ValueError("steps must be positive")
    previous = {path: path.stat().st_mtime_ns for path in (scene / "gs" / "ply").glob("*.ply")}
    run(
        [
            python, Path(gsplat_dir) / "examples" / "simple_trainer.py", "default",
            "--data-dir", scene, "--data-factor", "1", "--result-dir", scene / "gs",
            "--max-steps", str(steps), "--save-ply", "--ply-steps", str(steps),
            "--eval-steps", str(steps), "--save-steps", str(steps),
            "--no-normalize-world-space",  # keep the COLMAP frame so splats and cameras agree
            "--test-every", "8", "--disable-viewer", "--disable-video",
        ],
        cwd=Path(gsplat_dir) / "examples",
    )
    produced = [p for p in (scene / "gs" / "ply").glob("*.ply") if previous.get(p) != p.stat().st_mtime_ns]
    if not produced:
        raise FileNotFoundError("trainer returned without exporting a new PLY; refusing to reuse a stale export")
    return max(produced, key=lambda p: p.stat().st_mtime_ns)


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
    p.add_argument("--pose-confidence", type=float, default=5.0, help="VGGT depth confidence threshold (without BA)")
    p.add_argument("--shared-camera", action="store_true", help="share intrinsics for a capture made without zoom changes (BA)")
    p.add_argument("--ba-track-budget", type=int, default=32768, help="maximum frame-point pairs per tracking batch (BA)")
    p.add_argument("--ba-reprojection-error", type=float, default=8.0, help="initial reprojection inlier tolerance in pixels (BA)")
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
    p.add_argument("--pose-confidence", type=float, default=5.0, help="VGGT depth confidence threshold (without BA)")
    p.add_argument("--shared-camera", action="store_true", help="share intrinsics for a capture made without zoom changes (BA)")
    p.add_argument("--ba-track-budget", type=int, default=32768, help="maximum frame-point pairs per tracking batch (BA)")
    p.add_argument("--ba-reprojection-error", type=float, default=8.0, help="initial reprojection inlier tolerance in pixels (BA)")
    p.add_argument("--steps", type=int, default=7000)
    p.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS), help="movable things; empty = static world only")
    p.add_argument("--vggt-python", default=sys.executable)
    p.add_argument("--gsplat-python", default=sys.executable)
    p.add_argument("--sam3-python", default=sys.executable)
    p.add_argument("--report", type=Path, help="measured run JSON (default: SCENE/run.json)")
    p.add_argument("--start-at", choices=["frames", "poses", "train", "segment", "world"], default="frames",
                   help="explicitly reuse earlier stage outputs; skipped stages are not timed")
    world_args(p)

    p = sub.add_parser("demo")
    p.add_argument("--out", type=Path, default=Path("demo_world"))
    p.add_argument("--seed", type=int, default=0)

    a = ap.parse_args(argv)
    from .world import build_world  # numpy/scipy import deferred so --help is instant

    if a.cmd == "frames":
        extract_frames(a.video, a.scene, a.num, a.max_size)
    elif a.cmd == "poses":
        estimate_poses(a.scene, a.vggt_dir, a.ba, a.python, a.pose_confidence, a.shared_camera, a.ba_track_budget, a.ba_reprojection_error)
    elif a.cmd == "train":
        print(train_splats(a.scene, a.gsplat_dir, a.steps, a.python))
    elif a.cmd == "segment":
        segment_objects(a.scene, a.prompts, a.python)
    elif a.cmd == "world":
        masks = a.masks or ((a.scene / "masks") if (a.scene / "masks" / "labels.json").exists() else None)
        w = build_world(a.scene / "sparse", a.splats or latest_ply(a.scene), a.out, masks, a.eye_height, a.voxel)
        print(json.dumps(w["stats"], indent=1))
    elif a.cmd == "all":
        from .run_report import RunReport

        settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
        report = RunReport(a.report or a.scene / "run.json", a.video, settings)
        stages = ["frames", "poses", "train", "segment", "world"]

        def stage(name, fn, *args):
            if stages.index(name) < stages.index(a.start_at):
                report.data["stages"][name] = {"status": "reused", "seconds": None}
                report.save()
                return None
            return report.stage(name, fn, *args)

        stage("frames", extract_frames, a.video, a.scene, a.num, a.max_size)
        stage("poses", estimate_poses, a.scene, a.vggt_dir, a.ba, a.vggt_python, a.pose_confidence, a.shared_camera, a.ba_track_budget, a.ba_reprojection_error)
        ply = stage("train", train_splats, a.scene, a.gsplat_dir, a.steps, a.gsplat_python)
        if ply is None:
            ply = report.stage("validate_reused_splats", latest_ply, a.scene)
        masks = a.masks
        if masks is None and a.prompts.strip():
            stage("segment", segment_objects, a.scene, a.prompts, a.sam3_python)
            masks = a.scene / "masks"
        else:
            report.data["stages"]["segment"] = {"status": "provided" if masks else "disabled", "seconds": None}
        w = stage("world", build_world, a.scene / "sparse", ply, a.out, masks, a.eye_height, a.voxel)
        report.finish(w)
        print(json.dumps(report.data, indent=1))
    elif a.cmd == "demo":
        from .synthetic import make_scene

        make_scene(a.out / "_capture", seed=a.seed)
        w = build_world(a.out / "_capture" / "sparse", a.out / "_capture" / "splats.ply", a.out, a.out / "_capture" / "masks")
        print(json.dumps(w["stats"], indent=1))
        print(f"serve with: python -m http.server -d {a.out} 8000  ->  http://localhost:8000/")


if __name__ == "__main__":
    main()
