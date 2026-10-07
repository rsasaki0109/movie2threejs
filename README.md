# playworld (working name)

**Phone video → a walkable, physical 3D world in the browser.** Walk through your own room,
kick the chair over, throw things at the mugs. Built on three.js, gaussian splatting and Rapier.

> Status: pre-alpha. The geometry and the viewer are tested on a synthetic room; the GPU stages
> (VGGT, gsplat, SAM 3) are wired up but have not yet been run end to end on a real video.

## How it works

```
video ──frames──▶ VGGT (camera poses + points, COLMAP format)
                    │
                    ├─▶ gsplat (3D gaussian splatting)
                    ├─▶ SAM 3 (movable things by text prompt, tracked through the video)
                    ▼
               playworld world
                 • gravity from how the phone was held + RANSAC floor
                 • metric scale from eye height
                 • 2D masks lifted onto gaussians (z-buffer visibility, majority vote)
                 • each object → solid convex hull down to the surface it rests on
                 • walls and furniture → a few merged, hole-free boxes
                    ▼
               world.json + splats + viewer (three.js + Spark + Rapier)
```

## Try it

No GPU, synthetic room:

```bash
pip install -e .
playworld demo --out demo_world
python -m http.server -d demo_world 8000   # open http://localhost:8000
```

Your own video: open `notebooks/playworld_colab.ipynb` in Colab (GPU runtime).

Controls: **WASD** walk · **Space** jump · **Click** throw · **E** push · **R** reset · **C** colliders.

## Stages

| command | does | environment |
|---|---|---|
| `playworld frames VIDEO SCENE` | evenly sampled frames | ffmpeg |
| `playworld poses SCENE --vggt-dir` | VGGT feed-forward poses → `SCENE/sparse` | VGGT env |
| `playworld train SCENE --gsplat-dir` | gaussian splatting → `SCENE/gs/ply` | gsplat env |
| `playworld segment SCENE --prompts "chair,box"` | instance masks → `SCENE/masks` | SAM 3 env |
| `playworld world SCENE --out OUT` | world.json, split splats, viewer | numpy + scipy |
| `playworld all VIDEO SCENE --out OUT ...` | everything | per-stage `--*-python` |

VGGT, gsplat and SAM 3 pin incompatible numpy/torch versions, so each runs in its own environment.

## Known limitations

- Scale comes from an assumed eye height (`--eye-height`, default 1.5 m); expect a few percent error.
- Gravity assumes the phone was held roughly upright.
- Moving an object reveals a hole where it stood (nothing was ever seen there).
- Only surfaces the camera saw exist; objects are made solid by extending their hull to the surface below.
- Weights: the default VGGT checkpoint is non-commercial; SAM 3 weights are gated on Hugging Face.

## Development

```bash
pip install -e '.[dev]' && pytest
python notebooks/make_colab.py   # regenerate the notebook
```
