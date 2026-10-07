![Real public room capture becoming a walkable browser world, with a box pushed over](docs/hero.gif)

**Real room capture → a walkable, physical three.js world.**

[![Open Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
Upload [the notebook](notebooks/playworld_colab.ipynb) and the current project ZIP. There is no public repository to clone yet.

Try the verified synthetic room in three commands:

```bash
pip install -e .
playworld demo --out demo_world
python -m http.server -d demo_world 8000
```

Open http://localhost:8000. **WASD** walk · **Space** jump · **Click** throw · **E** push · **R** reset · **C** colliders.

# playworld (working name)

The hero uses a real public apartment capture from **Eyeful Tower (MIT)**, visualized from capture-rig photographs. It is not a smartphone recording or the author's room. [Source and license](docs/data-attribution.md) · [High-quality MP4](docs/hero.mp4) · [Measured GPU runs](docs/benchmarks.md).

VGGT, gsplat and SAM 3 ran on an existing Colab L4. The complete workbench reconstruction took **419.079 s** with cached weights, excluding setup. Subsequent measured refinements reuse its poses and Gaussians; the final world has **8 movable objects and 1,928 static colliders**. Browser checks confirm walking, a ray push that drops a box from the workbench, a thrown ball, and fixed-step replay. The 13 s hero is actual viewer output with a scripted camera and physics events.

## How it works

```
video → frames → VGGT camera poses + points
                   ├─ gsplat → Gaussian reconstruction
                   └─ SAM 3 → tracked object masks
                          ↓
                 floor, scale, rigid bodies + static colliders
                          ↓
                 three.js + Spark + Rapier → browser world
```

The world builder estimates gravity from upright camera poses, fits the lower supported floor and assumes a camera height for scale. It lifts masks onto visible Gaussians, separates movable objects, and makes each object solid down to its support. Compact background points become merged static collision boxes. Object spill is removed before export; small flat support patches and colored convex interiors cover unobserved surfaces approximately.

## Make a real capture

The [GPU recipe](docs/gpu.md) and notebook share the same isolated VGGT, gsplat and SAM 3 environments. The public workbench recipe uses 24 frames, 7,000 training steps and bundle adjustment. Stage reports preserve timings, failures, source hashes and actual GPU information. [Colab CLI instructions](docs/colab-cli.md) use an existing runtime.

For your own video, keep the phone upright, move slowly in a bright room, and capture the floor and the sides of objects. The pipeline accepts video input; a real smartphone capture has not been validated yet. Review floor height, masks and collisions before recording.

The [recording workflow](docs/recording.md) uses a JSON timeline, fixed 1/60 s physics and Playwright canvas capture. [The hero shot](shots/hero.json) records camera positions, a push and a throw; its action ray is specified independently of the cinematic camera. Interactive walking uses Rapier's capsule character controller.

## Stages

| command | output |
|---|---|
| `playworld frames VIDEO SCENE` | evenly sampled frames |
| `playworld poses SCENE --vggt-dir ...` | VGGT poses and points in COLMAP format |
| `playworld train SCENE --gsplat-dir ...` | trained Gaussians |
| `playworld segment SCENE --prompts "chair,box"` | tracked instance masks |
| `playworld world SCENE --out OUT` | world.json, split splats and viewer |
| `playworld all VIDEO SCENE --out OUT ...` | measured pipeline; per-stage Python environments |

## Known limitations

- Metric scale assumes a camera height (`--eye-height`, default 1.5 m); true scale accuracy is unmeasured. Gravity assumes an upright camera.
- Unseen surfaces lack photographic texture. Flat support patches and inward-facing colored convex hulls are simple approximations. Thin or open objects can look solid when turned over.
- The public workbench has blur and incomplete floor/edge coverage outside the observed viewpoints. There is no general room inpainting or automatic boundary completion.
- Masks, convex colliders and masses are estimates. Review them before using a new scene; not every detected label becomes a useful movable object.
- The default VGGT weights are non-commercial. SAM 3 weights require approved Hugging Face access; model terms apply separately from the dataset's MIT license.
- SAM 3 was verified on L4. T4 compatibility is unverified; the current object recipe requires native bf16 hardware.
- Shared GPU scripts were executed through Colab CLI. The notebook's interactive upload/preview UI was not automated.

## Development

```bash
pip install -e '.[dev]'
pytest
python notebooks/make_colab.py
node --test tests/record.test.mjs
```

For capture: `pip install -e '.[record]'` and `python -m playwright install chromium`.
