"""Synthetic room capture for testing the pipeline and the viewer without a GPU.

Produces what the GPU stages would: a COLMAP model and a splat PLY in an arbitrary rotated,
scaled frame (like a feed-forward reconstruction), plus per-frame instance masks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import objects, splat_io
from .colmap_io import Camera, Image, Reconstruction, rotmat_to_qvec, write_model


def _box_surface(rng, lo, hi, density):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    pts = []
    for axis in range(3):
        a, b = [i for i in range(3) if i != axis]
        area = (hi[a] - lo[a]) * (hi[b] - lo[b])
        n = max(int(area * density), 8)
        for side in (lo[axis], hi[axis]):
            p = rng.uniform(lo, hi, size=(n, 3))
            p[:, axis] = side
            pts.append(p)
    return np.concatenate(pts)


def _plane(rng, lo, hi, density):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    area = np.prod([d for d in hi - lo if d > 0])
    return rng.uniform(lo, hi, size=(int(area * density), 3))


def room(rng, density=2500):
    """Points, colors, labels (0 = static) and label names of a 6 x 5 m room, y-up, meters."""
    parts = []  # (points, rgb, label)
    floor = _plane(rng, [-3, 0, -2.5], [3, 0, 2.5], density)
    checker = ((np.floor(floor[:, 0] * 2) + np.floor(floor[:, 2] * 2)) % 2)[:, None]
    parts.append((floor, 0.35 + 0.25 * checker * np.array([1.0, 0.9, 0.7]), 0))
    walls = [
        ([-3, 0, -2.5], [3, 2.6, -2.5], [0.85, 0.82, 0.75]),
        ([-3, 0, 2.5], [3, 2.6, 2.5], [0.80, 0.84, 0.86]),
        ([-3, 0, -2.5], [-3, 2.6, 2.5], [0.78, 0.80, 0.70]),
        ([3, 0, -2.5], [3, 2.6, 2.5], [0.70, 0.75, 0.82]),
    ]
    for lo, hi, c in walls:
        p = _plane(rng, lo, hi, density * 0.6)
        parts.append((p, np.tile(c, (len(p), 1)), 0))
    table = _box_surface(rng, [0.8, 0.0, -2.2], [2.2, 0.75, -1.4], density)
    parts.append((table, np.tile([0.45, 0.30, 0.18], (len(table), 1)), 0))
    names = {1: "crate", 2: "mug", 3: "chair"}
    crate = _box_surface(rng, [-1.6, 0.0, -1.0], [-1.1, 0.5, -0.5], density * 2)
    parts.append((crate, np.tile([0.85, 0.55, 0.15], (len(crate), 1)), 1))
    mug = _box_surface(rng, [1.3, 0.75, -1.9], [1.42, 0.87, -1.78], density * 20)
    parts.append((mug, np.tile([0.9, 0.15, 0.2], (len(mug), 1)), 2))
    seat = _box_surface(rng, [-0.3, 0.0, 1.2], [0.2, 0.9, 1.7], density * 2)
    parts.append((seat, np.tile([0.2, 0.45, 0.8], (len(seat), 1)), 3))
    pts = np.concatenate([p for p, _, _ in parts])
    rgb = np.concatenate([c for _, c, _ in parts]).clip(0, 1)
    lab = np.concatenate([np.full(len(p), l) for p, _, l in parts]).astype(np.int32)
    return pts, rgb, lab, names


def _look_rotation(forward, up=np.array([0.0, 1.0, 0.0])):
    """World-to-camera rotation (OpenCV: x right, y down, z forward) for a y-up world."""
    z = forward / np.linalg.norm(forward)
    x = np.cross(-up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def capture(rng, num_views=24, width=320, height=240, eye=1.55):
    views = []
    f = 0.9 * width
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    for i in range(num_views):
        a = 2 * np.pi * i / num_views
        # walk a small loop near the middle of the room while panning around, camera tilted down a bit
        c = np.array([0.5 * np.cos(a), eye + rng.normal(0, 0.04), 0.4 * np.sin(a)])
        fwd = np.array([np.cos(a), -0.5 + rng.normal(0, 0.05), np.sin(a)])
        R = _look_rotation(fwd)
        tilt = rng.normal(0, np.deg2rad(4))  # handheld roll
        Rz = np.array([[np.cos(tilt), -np.sin(tilt), 0], [np.sin(tilt), np.cos(tilt), 0], [0, 0, 1]])
        R = Rz @ R
        views.append(objects.View(K, R, -R @ c, width, height))
    return views


def render_masks(points, labels, views, cell=2):
    masks = []
    for v in views:
        vis, ui, vi = objects.visible(points, v, cell=cell)
        m = np.zeros((v.height, v.width), dtype=np.int32)
        idx = np.flatnonzero(vis & (labels > 0))
        for dv in range(cell):
            for du in range(cell):
                m[np.clip(vi[idx] // cell * cell + dv, 0, v.height - 1), np.clip(ui[idx] // cell * cell + du, 0, v.width - 1)] = labels[idx]
        masks.append(m)
    return masks


def random_similarity(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return float(rng.uniform(0.2, 3.0)), R, rng.normal(0, 2, size=3)


def make_scene(out: Path, seed: int = 0, num_views: int = 24) -> dict:
    """Writes out/sparse, out/splats.ply, out/masks; returns ground-truth info."""
    rng = np.random.default_rng(seed)
    out = Path(out)
    pts, rgb, lab, names = room(rng)
    views = capture(rng, num_views)
    masks = render_masks(pts, lab, views)

    # Move everything into an arbitrary reconstruction frame: p_rec = s R p + t.
    s, Rg, tg = random_similarity(rng)
    pts_rec = s * pts @ Rg.T + tg
    cams = {1: Camera(1, "PINHOLE", views[0].width, views[0].height, np.array([views[0].K[0, 0], views[0].K[1, 1], views[0].K[0, 2], views[0].K[1, 2]]))}
    images = {}
    for i, v in enumerate(views):
        # world-to-camera in the new frame: x_c = R (p - c) with p = (p_rec - t) / s / Rg
        R_new = v.R @ Rg.T
        c_world = -v.R.T @ v.t
        c_rec = s * Rg @ c_world + tg
        images[i + 1] = Image(i + 1, rotmat_to_qvec(R_new), -R_new @ c_rec, 1, f"frame_{i:04d}.png")
    sub = rng.choice(len(pts_rec), size=min(len(pts_rec), 40000), replace=False)
    write_model(Reconstruction(cams, images, pts_rec[sub], (rgb[sub] * 255).astype(np.uint8)), out / "sparse")
    splat_io.write_ply(out / "splats.ply", splat_io.make_splats(pts_rec, rgb, scale=0.025 * s))
    (out / "masks").mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(masks):
        np.save(out / "masks" / f"frame_{i:04d}.npy", m)
    (out / "masks" / "labels.json").write_text(json.dumps({str(k): v for k, v in names.items()}))
    return {"scale": s, "R": Rg, "t": tg, "names": names, "labels": lab, "points": pts}
