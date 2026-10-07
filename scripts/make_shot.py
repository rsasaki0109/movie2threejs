"""Generate a 13s starting shot from world coordinates; inspect and tune before hero capture."""
import argparse
import json
from pathlib import Path


def make_shot(world: dict) -> dict:
    objects = world["objects"]
    floor_objects = [o for o in objects if o["support_y"] < 0.1]
    if not floor_objects:
        raise ValueError("a floor object is needed for the push shot")
    subject = max(floor_objects, key=lambda o: max(p[1] for p in o["hull"]) - min(p[1] for p in o["hull"]))
    x, y, z = subject["centroid"]
    height = world["player"]["eye_height"]
    sx, _, sz = world["player"]["spawn"]
    start = [sx, height, sz]
    near = [x, height, z + 1.8]
    target = [x, y + 0.15, z]
    keys = [
        {"time": 0, "position": start, "target": target},
        {"time": 2, "position": start, "target": target},
        {"time": 5, "position": near, "target": target},
        {"time": 7.5, "position": near, "target": target},
    ]
    events = [{"time": 6, "type": "push"}]
    tabletop = [o for o in objects if o["support_y"] >= .1]
    if tabletop:
        mug = min(tabletop, key=lambda o: o["mass"])
        mx, my, mz = mug["centroid"]
        keys += [
            {"time": 8.3, "position": [mx, height, mz + 1.5], "target": [mx, my + .15, mz]},
            {"time": 10.5, "position": [mx, height, mz + 1.5], "target": [mx, my + .15, mz]},
        ]
        events.append({"time": 8.7, "type": "throw"})
    keys += [
        {"time": 12.5, "position": [x + 1.5, height + .7, z + 3.2], "target": target},
        {"time": 13, "position": start, "target": target},
    ]
    return {"version": 1, "fps": 30, "duration": 13, "camera": keys, "events": events}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    world = json.loads((args.world / "world.json").read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(make_shot(world), indent=2) + "\n", encoding="utf-8")
    print(f"{args.out}: provisional camera positions; check walls, framing, push and throw hits")
