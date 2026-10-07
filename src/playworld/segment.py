"""Per-frame instance masks of movable things with SAM 3 (text prompts, tracked through the video).

Writes scene/masks/<frame stem>.npy (int32, 0 = static world) and scene/masks/labels.json.
Runs in its own process so its GPU memory is released before the next stage.

Requires: pip install git+https://github.com/facebookresearch/sam3.git and access to the gated
facebook/sam3 checkpoint on Hugging Face (`hf auth login`).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

DEFAULT_PROMPTS = ["chair", "box", "cup", "mug", "bottle", "ball", "plant pot", "lamp", "backpack", "book", "pillow", "stool"]


def segment(scene: Path, prompts: list[str], min_area: float = 0.0005) -> dict[int, str]:
    from PIL import Image
    from sam3.model_builder import build_sam3_video_predictor

    frames = sorted((scene / "images").glob("*"))
    if not frames:
        raise FileNotFoundError(f"no frames in {scene / 'images'}")
    jpg_dir = scene / "_sam3_frames"  # SAM 3 reads a folder of <index>.jpg
    shutil.rmtree(jpg_dir, ignore_errors=True)
    jpg_dir.mkdir()
    for i, f in enumerate(frames):
        Image.open(f).convert("RGB").save(jpg_dir / f"{i}.jpg", quality=95)
    w, h = Image.open(frames[0]).size

    masks = np.zeros((len(frames), h, w), dtype=np.int32)
    names: dict[int, str] = {}
    predictor = build_sam3_video_predictor()
    session = predictor.handle_request(dict(type="start_session", resource_path=str(jpg_dir)))["session_id"]
    for prompt in prompts:
        predictor.handle_request(dict(type="reset_session", session_id=session))
        predictor.handle_request(dict(type="add_prompt", session_id=session, frame_index=0, text=prompt))
        ids: dict[int, int] = {}  # SAM 3 object id -> playworld label
        for resp in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session)):
            fi, out = resp["frame_index"], resp["outputs"]
            for obj_id, m in zip(np.asarray(out["out_obj_ids"]).tolist(), np.asarray(out["out_binary_masks"])):
                if m.shape != (h, w) or m.mean() < min_area:
                    continue
                if obj_id not in ids:
                    ids[obj_id] = len(names) + 1
                    names[ids[obj_id]] = prompt if prompt not in names.values() else f"{prompt} {sum(n.startswith(prompt) for n in names.values()) + 1}"
                free = masks[fi] == 0  # earlier prompts win overlaps
                masks[fi][m.astype(bool) & free] = ids[obj_id]
        print(f"{prompt!r}: {len(ids)} objects")
    try:
        predictor.handle_request(dict(type="close_session", session_id=session))
    except Exception:  # older/newer SAM 3 builds may name this differently; the process exits anyway
        pass
    shutil.rmtree(jpg_dir, ignore_errors=True)

    out_dir = scene / "masks"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir()
    for f, m in zip(frames, masks):
        np.save(out_dir / (f.stem + ".npy"), m)
    (out_dir / "labels.json").write_text(json.dumps({str(k): v for k, v in names.items()}, indent=1))
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS), help="comma-separated things that should be movable")
    a = ap.parse_args()
    names = segment(a.scene, [p.strip() for p in a.prompts.split(",") if p.strip()])
    print(json.dumps(names, indent=1))


if __name__ == "__main__":
    main()
