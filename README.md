# playworld (working name)

<!-- Add docs/hero.gif here only after the real-data reconstruction and recording pass. -->

**Walk around and knock things over in your browser.** A room-capture prototype built with three.js, Spark and Rapier.

[![Open Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
Upload [the notebook](notebooks/playworld_colab.ipynb) and the current project ZIP; no public repository is available yet.

## Quick start

From the project directory, try the verified synthetic room with three commands:

```bash
pip install -e .
playworld demo --out demo_world
python -m http.server -d demo_world 8000
```

Open http://localhost:8000. **WASD** walk · **Space** jump · **Click** throw · **E** push · **R** reset · **C** colliders.

> Pre-alpha: geometry, browser rendering, walking, pushing, throwing and repeatable capture
> are verified on a synthetic room. VGGT, gsplat and SAM 3 completed an Eyeful Tower
> apartment capture on a Colab L4 in 273.476 s (setup excluded), exporting four objects.
> Its first reconstruction has substantial blur; real-data quality and physical interaction
> are still being reviewed. The hero GIF/MP4 are pending. See [measurements](docs/benchmarks.md).

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

## Real captures and recording

The [GPU recipe](docs/gpu.md) uses isolated environments and saves stage timings, failures,
GPU information and source hashes. Colab runs the same setup/run scripts as a Linux host.
For the public-data demo, the notebook downloads an Eyeful Tower apartment capture with
its MIT notice. This is a capture-rig photograph sequence, not a smartphone video.
See [data attribution](docs/data-attribution.md).

The [recording workflow](docs/recording.md) uses a JSON camera/event timeline, fixed 1/60 s
physics and Playwright frame capture. The encoder targets a 13 s, 960px looping GIF below
8 MB and a 1920×1080 MP4. Those are output targets, not completed real-data artifacts.

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

- Scale comes from an assumed phone/camera height (`--eye-height`, default 1.5 m); real-data accuracy is unmeasured.
- Gravity assumes the phone was held roughly upright.
- Unseen surfaces are missing. Small flat patches use nearby support-surface colors to cover
  object footprints when samples exist; this approximation is tested only on synthetic data.
- Floater removal, room boundary cropping, real object masks and collision quality still need real-data review.
- Only surfaces the camera saw exist; objects are made solid by extending their hull to the surface below.
- Weights: the default VGGT checkpoint is non-commercial; SAM 3 weights are gated on Hugging Face.
- T4/SAM 3 compatibility is not verified. The current GPU recipe requires native bf16 hardware for movable objects.

## Development

```bash
pip install -e '.[dev]' && pytest
python notebooks/make_colab.py   # regenerate the notebook
node --test tests/record.test.mjs
python scripts/verify_viewer.py # after installing .[record], Chromium and generating demo_world
```
