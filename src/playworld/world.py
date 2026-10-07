"""Assemble a playable world: align the capture, split objects, build colliders, write world.json."""

from __future__ import annotations

import json
import shutil
from importlib import resources
from pathlib import Path

import numpy as np

from . import colliders, gravity, objects, splat_io
from .colmap_io import Reconstruction, read_model

FORMAT = "playworld/1"


def load_masks(masks_dir: Path, image_names: list[str]) -> tuple[list[np.ndarray], dict[int, str]]:
    """Masks are stored as <image stem>.npy integer arrays plus labels.json {"1": "chair", ...}."""
    masks_dir = Path(masks_dir)
    masks = [np.load(masks_dir / (Path(n).stem + ".npy")) for n in image_names]
    names = {int(k): v for k, v in json.loads((masks_dir / "labels.json").read_text()).items()}
    return masks, names


def views_for(rec: Reconstruction, masks: list[np.ndarray] | None = None) -> list[objects.View]:
    views = []
    for i, im in enumerate(rec.sorted_images()):
        cam = rec.cameras[im.camera_id]
        K, w, h = cam.K.copy(), cam.width, cam.height
        if masks is not None and masks[i].shape != (h, w):
            mh, mw = masks[i].shape
            K[0] *= mw / w
            K[1] *= mh / h
            w, h = mw, mh
        views.append(objects.View(K, im.R, im.tvec, w, h))
    return views


def build_world(
    sparse_dir: Path,
    splat_ply: Path,
    out_dir: Path,
    masks_dir: Path | None = None,
    eye_height: float = 1.5,
    voxel: float = 0.1,
    density: float = 400.0,
    copy_viewer: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = read_model(sparse_dir)
    images = rec.sorted_images()
    rotations = np.stack([im.R for im in images])
    centers = np.stack([im.center for im in images])

    down = gravity.down_from_cameras(rotations)
    floor = gravity.fit_floor(rec.xyz, centers, down)
    T = gravity.world_alignment(floor, centers, images[0].R, eye_height)

    splats = splat_io.read_ply(splat_ply)
    g_means = splat_io.means(splats)
    labels = np.zeros(len(g_means), dtype=np.int32)
    point_labels = np.zeros(len(rec.xyz), dtype=np.int32)
    names: dict[int, str] = {}
    if masks_dir is not None:
        masks, names = load_masks(masks_dir, [im.name for im in images])
        views = views_for(rec, masks)
        solid = splat_io.opacity(splats) > 0.1
        labels[solid] = objects.lift_labels(g_means[solid], views, masks)
        point_labels = objects.lift_labels(rec.xyz, views, masks)

    g_world = gravity.apply(T, g_means)
    p_world = gravity.apply(T, rec.xyz)
    collider_labels = labels.copy()
    world_objects = []
    for label in sorted(set(np.unique(labels)) - {0}):
        sel = labels == label
        support = objects.support_height(g_world[sel], p_world[point_labels == 0])
        obj = objects.rigid_object(g_world[sel], int(label), names.get(int(label), f"object{label}"), support)
        if obj is None:
            labels[sel] = 0
            continue
        # Whatever lies inside the solid hull belongs to the object (unlabeled sides, stray splats),
        # otherwise it would stay behind as a ghost and as a static collider inside the body.
        lo = obj.centroid + obj.hull.min(axis=0) - 0.05
        hi = obj.centroid + obj.hull.max(axis=0) + 0.05
        near = np.flatnonzero((labels == 0) & np.all((g_world > lo) & (g_world < hi), axis=1))
        labels[near[objects.inside_hull(g_world[near], obj)]] = label
        # Static collision keeps a wider berth so bodies are not wedged by leftovers around them.
        lo, hi = lo - 0.1, hi + 0.1
        for arr, lab in ((p_world, point_labels), (g_world, collider_labels)):
            near = np.flatnonzero((lab == 0) & np.all((arr > lo) & (arr < hi), axis=1))
            lab[near[objects.inside_hull(arr[near], obj, grow=0.08)]] = label
        sel = labels == label
        rel = f"objects/{label}.ply"
        splat_io.write_ply(out_dir / rel, splat_io.subset(splats, sel))
        world_objects.append(obj.to_json(rel, density) | {"support_y": round(support, 3)})
    splat_io.write_ply(out_dir / "background.ply", splat_io.subset(splats, labels == 0))

    # Collision points: the sparse reconstruction plus solid, compact gaussians (much denser).
    solid = (collider_labels == 0) & (splat_io.opacity(splats) > 0.5) & (splat_io.max_scale(splats) * np.linalg.norm(T[:3, 0]) < 0.1)
    static_pts = np.vstack([p_world[point_labels == 0], g_world[solid]])
    boxes = colliders.static_boxes(static_pts, voxel=voxel)
    for o in world_objects:
        h = np.array(o["hull"]) + o["centroid"]
        boxes = colliders.drop_overlapping(boxes, h.min(axis=0) - 0.05, h.max(axis=0) + 0.05)
    boxes = [colliders.floor_box(static_pts)] + boxes

    spawn = gravity.apply(T, centers[:1])[0]
    world = {
        "format": FORMAT,
        "units": "meters",
        "up": "+y",
        "align": [round(float(v), 8) for v in T.T.reshape(-1)],  # column-major, as three.js expects
        "background": "background.ply",
        "player": {"spawn": [round(float(spawn[0]), 3), 0.0, round(float(spawn[2]), 3)], "eye_height": eye_height},
        "colliders": [b.to_json() for b in boxes],
        "objects": world_objects,
        "stats": {
            "cameras": len(images),
            "points": int(len(rec.xyz)),
            "gaussians": int(len(g_means)),
            "floor_inliers": int(floor.inliers.sum()),
            "scale_m_per_unit": round(float(np.linalg.norm(T[:3, 0])), 6),
        },
    }
    (out_dir / "world.json").write_text(json.dumps(world, indent=1))
    if copy_viewer:
        install_viewer(out_dir)
    return world


def install_viewer(out_dir: Path) -> None:
    for name in ("index.html", "main.js"):
        with resources.as_file(resources.files("playworld.viewer") / name) as src:
            shutil.copy(src, Path(out_dir) / name)
