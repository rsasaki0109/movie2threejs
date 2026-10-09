"""Assemble a playable world: align the capture, split objects, build colliders, write world.json."""

from __future__ import annotations

import json
import shutil
from importlib import resources
from pathlib import Path

import numpy as np

from . import colliders, gravity, objects, splat_io
from .support import surface_patch
from .cleanup import surface_splat_mask
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
    bounds_margin: float | None = None,
    clean_splats: bool = False,
    object_filter: str = "mad",
) -> dict:
    if object_filter not in {"mad", "connected"}:
        raise ValueError(f"unknown object filter: {object_filter}")
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
    within_bounds = np.ones(len(g_world), dtype=bool)
    if bounds_margin is not None:
        aligned_cameras = gravity.apply(T, centers)
        within_bounds = colliders.near_capture(g_world, aligned_cameras, bounds_margin)
        labels[~within_bounds] = -1
        point_labels[~colliders.near_capture(p_world, aligned_cameras, bounds_margin)] = -1
    world_scale = np.linalg.norm(T[:3, 0])
    sizes = splat_io.max_scale(splats) * world_scale
    discarded_diffuse = np.zeros(len(g_world), dtype=bool)
    if clean_splats:
        scales = np.exp(np.stack([splats[f"scale_{i}"] for i in range(3)], axis=1)) * world_scale
        discarded_diffuse = (labels == 0) & ~surface_splat_mask(scales, splat_io.opacity(splats))
        labels[discarded_diffuse] = -1
    # A rigid body must not carry distant or very broad segmentation spill.
    # Discard those splats rather than leaving a static ghost at its old pose.
    for label in sorted(set(np.unique(labels)) - {0, -1}):
        ids = np.flatnonzero(labels == label)
        clean = objects.movable_splat_mask(g_world[ids], sizes[ids], method=object_filter)
        if clean.sum() >= 8:
            labels[ids[~clean]] = -1
    collider_labels = labels.copy()
    # Sparse tracks often miss an untextured tabletop. Use compact opaque
    # background Gaussians too, while excluding objects and diffuse floaters.
    compact = (labels == 0) & (splat_io.opacity(splats) > 0.5) & (splat_io.max_scale(splats) * np.linalg.norm(T[:3, 0]) < 0.1)
    support_points = np.vstack([p_world[point_labels == 0], g_world[compact]])
    rgb = np.stack([splats[f"f_dc_{i}"] for i in range(3)], axis=1) * splat_io.SH_C0 + 0.5
    world_objects = []
    for label in sorted(set(np.unique(labels)) - {0, -1}):
        sel = labels == label
        obj_points = g_world[sel]
        support = objects.support_height(obj_points[objects.inlier_mask(obj_points, method=object_filter)], support_points)
        obj = objects.rigid_object(g_world[sel], int(label), names.get(int(label), f"object{label}"), support, method=object_filter)
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
        world_objects.append(obj.to_json(rel, density) | {
            "support_y": round(support, 3),
            "fill_color": np.median(rgb[sel], axis=0).clip(0, 1).round(5).tolist(),
        })
    splat_io.write_ply(out_dir / "background.ply", splat_io.subset(splats, labels == 0))

    # Patches remain in the room when a body moves. Do not add them to collisions
    # or the captured Gaussian count; they are explicitly approximate surfaces.
    background = labels == 0
    horizontal = background & splat_io.horizontal_mask(splats, T) & (splat_io.opacity(splats) > 0.5)
    color_samples = horizontal if horizontal.sum() >= 3 else background
    patches = []
    for obj in world_objects:
        footprint = np.array(obj["hull"]) + obj["centroid"]
        patch = surface_patch(footprint, obj["support_y"], g_world[color_samples], rgb[color_samples], ring=0.3)
        if patch is not None:
            patches.append({"object_id": obj["id"], **patch})

    # Collision points: the sparse reconstruction plus solid, compact gaussians (much denser).
    solid = (collider_labels == 0) & (splat_io.opacity(splats) > 0.5) & (splat_io.max_scale(splats) * np.linalg.norm(T[:3, 0]) < 0.1)
    static_pts = np.vstack([p_world[point_labels == 0], g_world[solid]])
    boxes = colliders.static_boxes(static_pts, voxel=voxel)
    for o in world_objects:
        h = np.array(o["hull"]) + o["centroid"]
        boxes = colliders.carve_boxes(boxes, h.min(axis=0) - 0.08, h.max(axis=0) + 0.08)
    for o in world_objects:
        if o['support_y'] > 0.05:
            h = np.array(o['hull']) + o['centroid']
            boxes.append(colliders.support_box(h, o['support_y']))
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
        "support_patches": patches,
        "stats": {
            "cameras": len(images),
            "points": int(len(rec.xyz)),
            "gaussians": int((labels >= 0).sum()),
            "input_gaussians": int(len(g_means)),
            "discarded_object_gaussians": int(((labels < 0) & within_bounds & ~discarded_diffuse).sum()),
            "discarded_diffuse_gaussians": int(discarded_diffuse.sum()),
            "clean_splats": clean_splats,
            "object_filter": object_filter,
            "discarded_bounds_gaussians": int((~within_bounds).sum()),
            "bounds_margin_m": bounds_margin,
            "objects": len(world_objects),
            "colliders": len(boxes),
            "support_patches": len(patches),
            "floor_inliers": int(floor.inliers.sum()),
            "scale_m_per_unit": round(float(np.linalg.norm(T[:3, 0])), 6),
        },
    }
    (out_dir / "world.json").write_text(json.dumps(world, indent=1))
    if copy_viewer:
        install_viewer(out_dir)
    return world


def install_viewer(out_dir: Path) -> None:
    for name in ("index.html", "main.js", "record.js"):
        with resources.as_file(resources.files("playworld.viewer") / name) as src:
            shutil.copy(src, Path(out_dir) / name)
