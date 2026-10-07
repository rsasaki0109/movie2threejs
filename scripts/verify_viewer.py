"""Browser integration check: fixed-step replay, ray push, thrown ball, walking and pixels."""
import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    root = Path(__file__).resolve().parents[1]
    world = root / "demo_world"
    shot = json.loads((root / "shots/demo.json").read_text())
    # Fast test: each output frame still simulates its full fixed-step interval.
    shot["fps"] = 15
    shot["duration"] = 2
    position = shot["camera"][2]["position"]
    target = shot["camera"][2]["target"]
    shot["camera"] = [{"time": t, "position": position, "target": target} for t in [0, 2]]
    shot["events"] = [{"time": .2, "type": "push"}, {"time": .6, "type": "throw"},
                      {"time": .8, "type": "walk", "keys": ["KeyW"]},
                      {"time": 1.2, "type": "walk", "keys": []}]
    shot_bytes = json.dumps(shot).encode()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/test-shot.json":
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.end_headers(); self.wfile.write(shot_bytes)
            else:
                super().do_GET()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(world)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    runs, pictures = [], []
    out = root / ".cache/viewer-verification"
    out.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            for repeat in range(2):
                page = browser.new_page(viewport={"width": 640, "height": 360}, device_scale_factor=1)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/?record=/test-shot.json")
                page.wait_for_function("window.playworld?.ready || window.playworldError", polling=100, timeout=120_000)
                assert page.evaluate("window.playworldError") is None
                snapshots = []
                for frame in range(30):
                    snapshots.append(page.evaluate("playworld.record.nextFrame()"))
                    if frame in [0, 29]:
                        page.locator("canvas").screenshot(path=str(out / f"repeat-{repeat}-frame-{frame}.png"))
                assert not errors, errors
                pictures.append(page.locator("canvas").screenshot())
                runs.append(snapshots)
                page.close()
                print(f"replay {repeat + 1}/2 complete", flush=True)
            browser.close()
        assert runs[0] == runs[1], "physics replay differs"
        assert pictures[0] == pictures[1], "final frame pixels differ"
        chair = next(o for o in json.loads((world / "world.json").read_text())["objects"] if o["name"] == "chair")
        before = next(o for o in runs[0][0]["objects"] if o["id"] == chair["id"])
        after = next(o for o in runs[0][-1]["objects"] if o["id"] == chair["id"])
        distance = sum((after["position"][axis] - before["position"][axis])**2 for axis in "xyz")**.5
        tilt = (after["rotation"]["x"]**2 + after["rotation"]["z"]**2)**.5
        player_distance = sum((runs[0][-1]["player"][axis] - runs[0][0]["player"][axis])**2 for axis in "xz")**.5
        assert distance > .2, f"push missed (distance {distance})"
        assert tilt > .2, f"chair did not tip (tilt {tilt})"
        assert len(runs[0][-1]["balls"]) == 1, "throw event did not create a ball"
        assert player_distance > .1, "character did not walk"
        report = {"physics_replay_equal": True, "final_pixels_equal": True,
                  "chair_displacement_m": distance, "chair_quaternion_tilt": tilt,
                  "player_displacement_m": player_distance, "thrown_balls": 1,
                  "scope": "synthetic room only", "runs": runs}
        (out / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))
    finally:
        server.shutdown(); server.server_close()


if __name__ == "__main__":
    main()
