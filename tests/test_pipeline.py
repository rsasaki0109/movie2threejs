import json

import numpy as np
import pytest

from playworld import colliders, gravity, objects, splat_io
from playworld.colmap_io import qvec_to_rotmat, read_model, rotmat_to_qvec
from playworld.synthetic import make_scene, random_similarity
from playworld.world import build_world, views_for


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("scene")
    truth = make_scene(root / "capture", seed=3)
    world = build_world(root / "capture" / "sparse", root / "capture" / "splats.ply", root / "out", root / "capture" / "masks")
    return root, truth, world


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        _, R, _ = random_similarity(rng)
        assert np.allclose(qvec_to_rotmat(rotmat_to_qvec(R)), R, atol=1e-9)


def test_colmap_roundtrip(built):
    root, _, _ = built
    rec = read_model(root / "capture" / "sparse")
    assert len(rec.images) == 24 and len(rec.xyz) > 1000
    assert rec.cameras[1].model == "PINHOLE"


def test_ply_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    props = splat_io.make_splats(rng.normal(size=(100, 3)), rng.uniform(size=(100, 3)), 0.01)
    splat_io.write_ply(tmp_path / "a.ply", props)
    back = splat_io.read_ply(tmp_path / "a.ply")
    assert list(back) == list(props)
    for k in props:
        assert np.allclose(back[k], props[k], atol=1e-6)


def test_world_alignment_recovers_floor_and_scale(built):
    root, truth, world = built
    T = np.array(world["align"]).reshape(4, 4).T
    pts_rec = truth["scale"] * truth["points"] @ truth["R"].T + truth["t"]
    w = gravity.apply(T, pts_rec)
    floor = truth["points"][:, 1] == 0
    # floor flat at y=0 within 2 cm, i.e. gravity found and floor fitted
    assert np.abs(w[floor, 1]).max() < 0.02
    # true eye height 1.55 m, assumed 1.5 m: scale error ~3%
    ratio = np.linalg.norm(T[:3, 0]) * truth["scale"]
    assert ratio == pytest.approx(1.5 / 1.55, rel=0.03)
    # walls stand upright: wall height 2.6 m maps to ~2.5 m
    assert w[:, 1].max() == pytest.approx(2.6 * ratio, abs=0.05)


def test_objects_are_found(built):
    root, truth, world = built
    names = sorted(o["name"] for o in world["objects"])
    assert names == ["chair", "crate", "mug"]
    for o in world["objects"]:
        assert o["mass"] > 0
        assert (root / "out" / o["splat"]).exists()
    by = {o["name"]: o for o in world["objects"]}
    ratio = 1.5 / 1.55
    # the mug rests on the table top (0.75 m), crate and chair on the floor
    assert by["mug"]["support_y"] == pytest.approx(0.75 * ratio, abs=0.04)
    assert by["crate"]["support_y"] == 0.0 and by["chair"]["support_y"] == 0.0
    # solid volumes, not flat shells: 0.5 m crate ~ 0.11 m^3 * 400 kg/m^3
    assert by["crate"]["mass"] == pytest.approx(0.5**3 * ratio**3 * 400, rel=0.25)
    assert by["chair"]["mass"] == pytest.approx(0.5 * 0.9 * 0.5 * ratio**3 * 400, rel=0.25)


def test_object_splats_take_their_visible_gaussians(built):
    # Faces no camera ever saw do not exist in a real 3DGS reconstruction, so only count the visible ones.
    root, truth, world = built
    rec = read_model(root / "capture" / "sparse")
    pts_rec = truth["scale"] * truth["points"] @ truth["R"].T + truth["t"]
    seen = np.zeros(len(pts_rec), dtype=bool)
    for view in views_for(rec):
        seen |= objects.visible(pts_rec, view)[0]
    T = np.array(world["align"]).reshape(4, 4).T
    for o in world["objects"]:
        label = next(k for k, v in truth["names"].items() if v == o["name"])
        mine_seen = (truth["labels"] == label) & seen
        got = gravity.apply(T, splat_io.means(splat_io.read_ply(root / "out" / o["splat"])))
        n_true = int(mine_seen.sum())
        assert len(got) >= 0.9 * n_true, o["name"]
        assert len(got) <= 1.1 * int((truth["labels"] == label).sum()), o["name"]


def test_static_colliders_exclude_objects_and_are_merged(built):
    root, truth, world = built
    boxes = world["colliders"]
    assert 2 < len(boxes) < 2000  # merged, not one box per voxel
    floor = boxes[0]
    assert floor["center"][1] < 0
    # no static box where the crate stands (it is a dynamic body)
    crate = next(o for o in world["objects"] if o["name"] == "crate")
    cx, cy, cz = crate["centroid"]
    for b in boxes[1:]:
        c, h = np.array(b["center"]), np.array(b["half_extents"])
        assert not np.all(np.abs(np.array([cx, cy, cz]) - c) < h)


def test_world_json_and_viewer_written(built):
    root, _, world = built
    data = json.loads((root / "out" / "world.json").read_text())
    assert data["format"] == "playworld/1"
    assert (root / "out" / "index.html").exists() and (root / "out" / "main.js").exists()
    bg = splat_io.read_ply(root / "out" / "background.ply")
    total = sum(len(splat_io.read_ply(root / "out" / o["splat"])["x"]) for o in data["objects"]) + len(bg["x"])
    assert total == data["stats"]["gaussians"]


def test_greedy_boxes_cover_exactly():
    rng = np.random.default_rng(1)
    grid = rng.uniform(size=(12, 9, 10)) < 0.4
    covered = np.zeros_like(grid, dtype=int)
    for lo, hi in colliders.greedy_boxes(grid):
        covered[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += 1
    assert np.array_equal(covered, grid.astype(int))


def test_hidden_points_do_not_get_labels():
    # A wall point directly behind an object must stay background.
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1.0]])
    view = objects.View(K, np.eye(3), np.zeros(3), 100, 100)
    front = np.array([[0.0, 0.0, 1.0]] * 3)
    back = np.array([[0.0, 0.0, 3.0]] * 3)
    mask = np.zeros((100, 100), dtype=np.int32)
    mask[45:55, 45:55] = 1
    labels = objects.lift_labels(np.vstack([front, back]), [view, view], [mask, mask])
    assert labels.tolist() == [1, 1, 1, 0, 0, 0]


def _inside_any(p, boxes):
    return any(np.all(np.abs(p - np.array(b["center"])) <= np.array(b["half_extents"]) + 1e-6) for b in boxes)


def test_colliders_are_tight_and_closed(built):
    root, truth, world = built
    boxes = world["colliders"]
    ratio = 1.5 / 1.55
    # table top stays at table height (no voxel rounding upward)
    table_top = 0.75 * ratio
    tops = [b["center"][1] + b["half_extents"][1] for b in boxes[1:] if abs(b["center"][1] + b["half_extents"][1] - table_top) < 0.1]
    assert tops and max(tops) == pytest.approx(table_top, abs=0.02)
    # every sampled point of the walls at chest height lies inside a collider: no gaps to walk through
    T = np.array(world["align"]).reshape(4, 4).T
    pts_rec = truth["scale"] * truth["points"] @ truth["R"].T + truth["t"]
    w = gravity.apply(T, pts_rec)
    wall = (np.abs(np.abs(truth["points"][:, 0]) - 3) < 1e-9) & (np.abs(truth["points"][:, 1] - 1.2) < 0.3)
    sample = w[np.flatnonzero(wall)[::25]]
    inside = np.mean([_inside_any(p, boxes) for p in sample])
    assert inside > 0.98
