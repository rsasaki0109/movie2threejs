"""Capture fixed-step viewer frames with Playwright; simulation never waits on wall time."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright


def capture(world: Path, shot: Path, out: Path, width: int, height: int,
            software: bool = False, limit: int | None = None) -> dict:
    world, shot, out = world.resolve(), shot.resolve(), out.resolve()
    if not (world / "world.json").is_file():
        raise FileNotFoundError(world / "world.json")
    shot_data = shot.read_bytes()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("frame_*.png")):
        raise FileExistsError(f"{out} already has frames; choose a new output directory")

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/__shot.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(shot_data)))
                self.end_headers()
                self.wfile.write(shot_data)
            else:
                super().do_GET()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(world)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    started = time.perf_counter()
    snapshots, errors = [], []
    try:
        with sync_playwright() as p:
            args = ["--disable-background-timer-throttling", "--disable-renderer-backgrounding"]
            if software:
                args += ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
            else:
                args += ["--enable-gpu"]
                if sys.platform == "win32":
                    args += ["--use-gl=angle", "--use-angle=d3d11"]
            browser = p.chromium.launch(headless=True, args=args)
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            def page_error(error):
                errors.append(str(error))
                print(f"viewer error: {error}", flush=True)
            page.on("pageerror", page_error)
            page.on("requestfailed", lambda request: print(f"request failed: {request.url}: {request.failure}", flush=True))
            page.on("console", lambda message: print(f"browser: {message.text}", flush=True) if message.type == "error" else None)
            page.goto(f"http://127.0.0.1:{server.server_port}/?" + urlencode({"record": "/__shot.json"}))
            print("viewer loaded; waiting for splats and physics", flush=True)
            try:
                page.wait_for_function("window.playworld?.ready || window.playworldError", timeout=180_000, polling=100)
            except Exception:
                page.screenshot(path=str(out / "failed.png"))
                print("status:", page.locator("#status").inner_text(), "errors:", errors, flush=True)
                raise
            failure = page.evaluate("window.playworldError")
            if failure:
                raise RuntimeError(failure)
            assert page.evaluate("""() => [...document.querySelectorAll(
                'header, footer, #actions, #touch, #start, #hud, #cross')]
                .every(el => getComputedStyle(el).display === 'none')"""), "recording UI is visible"
            info = page.evaluate("""() => {
                const gl = playworld.renderer.getContext();
                const ext = gl.getExtension('WEBGL_debug_renderer_info');
                return {fps: playworld.record.fps, frames: playworld.record.frames,
                    photo_backdrops: (playworld.world.photo_backdrops ?? []).map(({texture,label}) => ({texture,label})),
                    renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)};
            }""")
            count = min(info["frames"], limit) if limit is not None else info["frames"]
            for index in range(count):
                snapshots.append(page.evaluate("playworld.record.nextFrame()"))
                page.locator("canvas").screenshot(path=str(out / f"frame_{index:05d}.png"), animations="disabled")
                if errors:
                    raise RuntimeError("\n".join(errors))
                if index % info["fps"] == 0:
                    print(f"captured {index + 1}/{count} frames", flush=True)
            browser.close()
        report = {
            **info, "captured_frames": count, "width": width, "height": height,
            "seconds": round(time.perf_counter() - started, 3),
            "world_sha256": hashlib.sha256((world / "world.json").read_bytes()).hexdigest(),
            "shot_sha256": hashlib.sha256(shot_data).hexdigest(),
            "shot": json.loads(shot_data), "snapshots": snapshots,
        }
        (out / "capture.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--shot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--software", action="store_true", help="use SwiftShader when no usable browser GPU exists")
    parser.add_argument("--limit", type=int, help="short smoke capture only; do not use for hero output")
    args = parser.parse_args()
    report = capture(args.world, args.shot, args.out, args.width, args.height, args.software, args.limit)
    print(json.dumps({k: v for k, v in report.items() if k not in ("snapshots", "shot")}, indent=2))


if __name__ == "__main__":
    main()
