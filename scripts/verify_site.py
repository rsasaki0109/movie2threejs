"""Check the real static world beneath a GitHub Pages-style path."""
from __future__ import annotations

import argparse
import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def verify(url: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    errors, failed, runs = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--enable-gpu", "--use-gl=angle", "--use-angle=d3d11"])
        for repeat in range(2):
            page = browser.new_page(viewport={"width": 960, "height": 540})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("requestfailed", lambda request: failed.append(request.url))
            page.goto(url + "?record=hero-shot.json")
            page.wait_for_function("window.playworld?.ready || window.playworldError", timeout=180_000)
            assert page.evaluate("window.playworldError") is None
            states = [page.evaluate("n => playworld.record.nextFrame(n)", n) for n in [0, 180, 270]]
            if repeat == 0:
                page.locator("canvas").screenshot(path=str(out / "physics.png"))
            runs.append(states)
            page.close()
            print(f"Real-room replay {repeat + 1}/2 complete", flush=True)
        assert runs[0] == runs[1], "physics differed between static-world replays"
        initial, final = runs[0][0], runs[0][-1]
        box_before = next(o for o in initial["objects"] if o["id"] == 6)
        box_after = next(o for o in final["objects"] if o["id"] == 6)
        drop = box_before["position"]["y"] - box_after["position"]["y"]
        assert drop > .5, "box did not fall off the workbench"
        assert any(a.get("object") == 6 for a in final["actions"]), "push missed the box"
        assert len(final["balls"]) == 1
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function("window.playworld?.ready || window.playworldError", timeout=180_000)
        assert page.evaluate("window.playworldError") is None
        page.screenshot(path=str(out / "desktop.png"))
        # Real user activation is required for pointer lock, so click the button.
        page.locator("#play").click()
        page.wait_for_function("playworld.controls.isLocked", timeout=15_000)
        before = page.evaluate("playworld.camera.position.toArray()")
        page.keyboard.down("KeyD")
        page.wait_for_timeout(700)
        page.keyboard.up("KeyD")
        after = page.evaluate("playworld.camera.position.toArray()")
        movement = sum((a - b)**2 for a, b in zip(after, before))**.5
        assert movement > .1, f"WASD movement failed: {movement}"
        page.keyboard.press("Escape")
        page.wait_for_function("!playworld.controls.isLocked")
        page.locator("#featured-push").click()
        assert page.locator("#featured-push").inner_text() == "Pushed!"
        page.locator("#reset-world").click()
        assert page.locator("#featured-push").inner_text() == "Knock the box over"
        page.close()
        context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, device_scale_factor=1)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function("window.playworld?.ready || window.playworldError", timeout=180_000)
        assert page.evaluate("window.playworldError") is None
        page.screenshot(path=str(out / "mobile-menu.png"))
        page.locator("#play").tap()
        assert page.locator("#touch").is_visible()
        session = context.new_cdp_session(page)
        forward = page.locator('[data-key="KeyW"]').bounding_box()
        point = {"x": forward["x"] + forward["width"] / 2,
                 "y": forward["y"] + forward["height"] / 2}
        before_touch = page.evaluate("playworld.camera.position.toArray()")
        session.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [point]})
        page.wait_for_timeout(600)
        session.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        after_touch = page.evaluate("playworld.camera.position.toArray()")
        touch_movement = sum((a - b)**2 for a, b in zip(after_touch, before_touch))**.5
        assert touch_movement > .1, f"touch movement failed: {touch_movement}"
        rotation_before = page.evaluate("playworld.camera.quaternion.toArray()")
        session.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": 170, "y": 300}]})
        session.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": 205, "y": 325}]})
        session.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        rotation_after = page.evaluate("playworld.camera.quaternion.toArray()")
        assert sum((a - b)**2 for a, b in zip(rotation_after, rotation_before)) > .001
        page.locator("#touch-throw").tap()
        page.screenshot(path=str(out / "mobile-play.png"))
        context.close()
        browser.close()
    assert not errors, errors
    assert not failed, failed
    report = {"url": url, "replay_equal": True, "box_drop_m": drop,
              "keyboard_walk_m": movement, "desktop_pointer_lock": True,
              "touch_entry_and_throw": True, "touch_walk_m": touch_movement,
              "touch_swipe_look": True, "mobile_scope": "Chromium emulation, not physical hardware",
              "page_errors": errors, "failed_requests": failed}
    (out / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Deployed base URL ending with /")
    parser.add_argument("--site", type=Path, default=ROOT / "_site")
    parser.add_argument("--out", type=Path, default=ROOT / ".cache/site-verification")
    args = parser.parse_args()
    if args.url:
        verify(args.url, args.out)
        return

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/movie2threejs/"):
                self.path = self.path[len("/movie2threejs"):]
                super().do_GET()
            else:
                self.send_error(404)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(args.site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        suffix = 'demos/workbench/' if (args.site/'demos/workbench/world.json').exists() else ''
        verify(f"http://127.0.0.1:{server.server_port}/movie2threejs/{suffix}", args.out)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
