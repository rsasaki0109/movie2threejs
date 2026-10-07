"""Combine your original phone video and captured world; encode MP4 and a <8 MB GIF."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_with_gifski(executable, mp4, gif, fps, quality, motion, lossy):
    producer = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(mp4),
        "-vf", f"fps={fps},scale=960:-2:flags=lanczos:out_color_matrix=bt709:out_range=tv",
        "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", "-",
    ], stdout=subprocess.PIPE)
    try:
        consumer = subprocess.Popen([
            str(executable), "--quiet", "--fps", str(fps), "--width", "960", "--quality", str(quality),
            "--motion-quality", str(motion), "--lossy-quality", str(lossy), "--repeat", "0",
            "--y4m-color-override", "bt709", "-o", str(gif), "-",
        ], stdin=producer.stdout)
        producer.stdout.close()
        if consumer.wait() != 0 or producer.wait() != 0:
            raise RuntimeError("gifski/ffmpeg stream failed")
    finally:
        if producer.poll() is None:
            producer.kill()
        producer.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--out", default=Path("docs"), type=Path)
    parser.add_argument("--video-start", type=float, default=0)
    parser.add_argument("--gifski", type=Path, help="optional gifski executable; otherwise use PATH or ffmpeg")
    args = parser.parse_args()
    capture = json.loads((args.frames / "capture.json").read_text())
    duration = capture["captured_frames"] / capture["fps"]
    if not 10 <= duration <= 15 or capture["captured_frames"] != capture["frames"]:
        raise ValueError("hero requires a complete capture lasting 10–15 seconds")
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    args.out.mkdir(parents=True, exist_ok=True)
    mp4, gif = args.out / "hero.mp4", args.out / "hero.gif"
    # The first two seconds show the source beside the actual world. The final
    # half-second dissolves to the first split frame for a continuous GIF loop.
    filters = (
        f"[0:v]fps={capture['fps']},scale=1920:1080,setsar=1,format=rgb24,split=2[all][intro];"
        "[intro]trim=duration=2,setpts=PTS-STARTPTS,scale=960:1080:force_original_aspect_ratio=decrease,"
        "pad=960:1080:(ow-iw)/2:(oh-ih)/2:color=0x111318[right];"
        f"[1:v]trim=duration=2,setpts=PTS-STARTPTS,fps={capture['fps']},"
        "scale=960:1080:force_original_aspect_ratio=decrease,"
        "format=rgb24,pad=960:1080:(ow-iw)/2:(oh-ih)/2:color=0x111318[left];"
        "[left][right]hstack=inputs=2[compare];"
        "[all]trim=start=2,setpts=PTS-STARTPTS[walk];"
        f"[compare][walk]concat=n=2:v=1:a=0,fps={capture['fps']},settb=AVTB,split=2[body][first];"
        f"[first]trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/({capture['fps']}*TB),"
        f"trim=duration={duration},format=rgba,fade=t=in:st={duration-0.5-1/capture['fps']:.6f}:d=0.5:alpha=1[still];"
        f"[body][still]overlay=shortest=1:format=rgb,trim=duration={duration},"
        "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[out]"
    )
    run("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", capture["fps"],
        "-i", args.frames / "frame_%05d.png", "-ss", args.video_start, "-i", args.video,
        "-filter_complex", filters, "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "17",
        "-preset", "slow", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-movflags", "+faststart", mp4)
    # Keep 960px. Reduce palette/fps only as needed; reject output over the limit.
    executable = args.gifski or shutil.which("gifski")
    colors, quality = None, None
    if executable:
        attempts = [(20, 90, 90, 90), (20, 80, 70, 70), (15, 70, 60, 60), (15, 60, 50, 50), (10, 40, 30, 30)]
        for fps, quality, motion, lossy in attempts:
            fps = min(fps, capture["fps"])
            encode_with_gifski(executable, mp4, gif, fps, quality, motion, lossy)
            if gif.stat().st_size <= 8_000_000:
                break
    else:
        attempts = [(20, 192), (20, 128), (15, 128), (15, 96), (15, 64), (10, 64), (9, 64), (8, 64)]
        for fps, colors in attempts:
            fps = min(fps, capture["fps"])
            palette = (
                f"fps={fps},scale=960:-2:flags=lanczos,split[a][b];"
                f"[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
            )
            run("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", mp4,
                "-filter_complex", palette, "-loop", "0", gif)
            if gif.stat().st_size <= 8_000_000:
                break
    if gif.stat().st_size > 8_000_000:
        raise RuntimeError("GIF exceeds 8 MB; adjust shot/capture before publishing")
    from PIL import Image
    with Image.open(gif) as animation:
        gif_duration = 0
        for index in range(animation.n_frames):
            animation.seek(index)
            gif_duration += animation.info.get("duration", 0) / 1000
        if animation.width != 960 or animation.info.get("loop") != 0 or not 10 <= gif_duration <= 15:
            raise RuntimeError("GIF dimensions, loop or measured duration do not meet hero requirements")
    report = {
        "source_video_sha256": sha256(args.video), "source_video_name": args.video.name,
        "capture_world_sha256": capture["world_sha256"], "shot_sha256": capture["shot_sha256"],
        "duration_seconds": duration, "gif_measured_seconds": round(gif_duration, 3),
        "gif_fps": fps, "palette_colors": colors, "quality": quality,
        "gif_encoder": "gifski" if executable else "ffmpeg",
        "gif_bytes": gif.stat().st_size, "mp4_bytes": mp4.stat().st_size,
    }
    (args.out / "hero-provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
