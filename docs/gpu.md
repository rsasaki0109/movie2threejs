# GPU execution

Use `notebooks/playworld_colab.ipynb` in Colab, or an already provisioned Linux GPU host.
No script creates a VM, Pod, repository, or paid resource. The same setup and run scripts
are used by the notebook and the shell recipe.

The [Colab CLI workflow](colab-cli.md) can transfer and execute the same scripts without
the notebook upload dialogs. Setup, GPU access, trainer help flags and SAM 3 API imports
passed on an existing Colab L4. The short public workbench reconstruction completed in
419.079 s with cached weights (setup excluded). Later object/world refinements reuse its
poses and trained Gaussians; see the separate [measured reports](benchmarks.md).

```bash
bash scripts/setup_gpu.sh
python scripts/download_public.py --out scenes/public_source --workshop
PLAYWORLD_POSE_CONFIDENCE=1.5 PLAYWORLD_PROMPTS='cardboard box,plastic bottle,folding chair' bash scripts/run.sh scenes/public_source/apartment-workshop.mp4 scenes/workshop world --ba --shared-camera --ba-reprojection-error 32.0
```

The setup creates three isolated GPU environments (VGGT, gsplat, SAM 3) plus a lightweight
core environment, pins upstream revisions, checks trainer help flags/imports, and saves
package versions. Linux, Python 3.12, ffmpeg, git, an NVIDIA GPU driver and the CUDA toolkit
(`nvcc`) are required. Set `PLAYWORLD_GPU_ROOT` to relocate environments and repositories.
Setup does not prove model inference works. On 2026-10-07, the Colab setup passed with
CUDA 13.0, PyTorch 2.9.1+cu130 for VGGT/gsplat and 2.10.0+cu130 for SAM 3. Gated weight
access and full model inference were confirmed from the runtime.
The setup selects PyTorch wheels matching the installed CUDA compiler (12.6, 12.8 or 13.0),
fetches gsplat's GLM submodule, and uses the noninteractive Matplotlib backend in its venvs.

The selected [gsplat v1.5.3 trainer](https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/examples/simple_trainer.py)
defines PLY export, disabling normalization, and disabling its extra trajectory video.
This version's examples use a fork of pycolmap with `SceneManager`; do not substitute the
official pycolmap bindings used in the VGGT environment. Evaluation holds out every eighth
frame instead of accidentally treating frame zero as the sole validation image.

The pinned [SAM 3 predictor](https://github.com/facebookresearch/sam3/blob/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc/sam3/model/sam3_base_predictor.py)
uses bf16 autocast. The run script requires native bf16 hardware (Ampere or newer) when
object prompts are enabled; T4 inference is not verified. Do not replace a physics demo
with a static world silently. To deliberately test static reconstruction only, use
`PLAYWORLD_PROMPTS=''`. SAM 3 also requires approved Hugging Face access to `facebook/sam3`;
set `HF_TOKEN` in Colab secrets or the host environment, never in committed files.

The historical video/VGGT workbench used 24 frames and 7,000 training steps. The
current README world uses provided camera calibration and a separately measured
[156-photograph showcase](quality.md#calibrated-public-showcase). These are not general VRAM
or speed guarantees. This BA reconstruction completed below the 15 minute target; see
[actual settings and quality limits](benchmarks.md). Optional settings:

```bash
PLAYWORLD_NUM_FRAMES=32 PLAYWORLD_TRAIN_STEPS=7000 bash scripts/run.sh room.mp4 scenes/room world
```

The public apartment video required `PLAYWORLD_POSE_CONFIDENCE=1.5`: on 24 frames,
VGGT's measured maximum confidence was 2.6803, so the upstream default threshold 5.0
exported zero points. The explicit 1.5 retry exported 100,000 finite points with 24 poses.
This is a measured setting for this capture, not a universal quality threshold. Inspect
the reconstruction before lowering it for other inputs. Empty/nonfinite exports now
fail the pose stage before starting gsplat. The CLI also accepts `--pose-confidence`.

After a failed stage, explicitly reuse earlier outputs:

```bash
bash scripts/run.sh room.mp4 scenes/room world --start-at segment
```

Each invocation writes `SCENE/run-<UTC>.json` and `.log`, plus the upstream revisions.
Failed stages retain measured time/error; reused stages have no fabricated time.
For the final benchmark, keep a full successful run and export its actual measurements:

```bash
python scripts/benchmark.py scenes/room/run-<UTC>.json --description 'Eyeful Tower public apartment capture; photograph sequence, not smartphone video'
```


## Local checkpoint refinement (tested on GTX 1660 Ti)

`scripts/setup_refine.sh` creates a separate Python 3.12 environment for refining
an existing SH3 gsplat checkpoint on Linux/WSL. It allocates no remote runtime.
It requires `uv`, an NVIDIA GPU exposed to Linux, a C++ compiler, and the CUDA
**12.8** toolkit (`nvcc`). This narrower recipe uses PyTorch 2.9.1+cu128 and the
same pinned gsplat 1.5.3 sources as the main setup. Follow the
[NVIDIA WSL installation guidance](https://docs.nvidia.com/cuda/archive/12.8.2/cuda-installation-guide-linux/index.html):
WSL uses the Windows GPU driver; install toolkit components rather than a Linux
display driver. The local verification installed `cuda-nvcc-12-8`,
`cuda-cudart-dev-12-8` and `cuda-cccl-12-8`. Native gsplat compilation took 10 min
17 s on this host, separate from refinement timing. That is not a setup-speed
guarantee.

```bash
bash scripts/setup_refine.sh
bash scripts/setup_refine.sh --check
CUDA_HOME=/usr/local/cuda-12.8 OMP_NUM_THREADS=4 "$HOME/.local/share/playworld/refine-gpu/bin/python" scripts/refine_checkpoint.py --scene scenes/room --checkpoint saved/ckpt_6999_rank0.pt --out refinement --steps 2000 --eval-every 500 --patch-size 2048 --depth-weight 0.5 --means-lr 0.00016
```

Choose a fresh output directory. `PLAYWORLD_REFINE_ENV` and
`PLAYWORLD_REFINE_REPO` relocate the environment and pinned source checkout.
`--scene` needs the original `images/` and `sparse/` camera/point model, not the
smaller segmentation scene. Images must already be undistorted PINHOLE images.
`--patch-size 2048` used full images in this experiment; the default 512 uses
random crops with correctly offset principal points. `--max-size 1118` limits
the longest image edge, adjusting each camera's intrinsics independently.

This is a **warm start with fresh Adam optimizers**, not an exact training
resume. Gaussian count and cameras stay fixed; no densification is performed.
The checkpoint must contain finite `means`, log `scales`, wxyz `quats`, logit
`opacities`, `sh0` and SH3 `shN` tensors. The experimental depth prior is disabled
by default. When enabled, it uses color-gated sparse SfM points from training
views only; sparse visibility can miss occluders, and these are not measured
depth maps. The weight and learning rate above were selected on this room's
existing validation split and are not universal defaults.

`refinement.json` records input/script hashes, versions, settings, per-image
PSNR/SSIM, elapsed time and peak allocated CUDA memory. `best.pt`/`best.ply`
contain the best refinement according to mean validation PSNR; if no step beats
the source, there is no `best` export. `latest` contains the final evaluated
step. Exports remain weights-only. Paired review images have the original
photograph on the left and rendered reconstruction on the right. To check a
saved checkpoint without optimizer steps, use `--evaluate-only` and another
fresh output directory.

The window-office trial refined 1,108,601 Gaussians with 126 training and 18
heldout photographs on a **GTX 1660 Ti, 6GB**. The selected 2,000-step run took
**361.575 s**, with **1,531.119 MiB peak allocated CUDA memory**. This does not
establish that the complete VGGT/SAM pipeline fits this GPU. Background blur
still failed visual review; see the [quality assessment](quality.md#local-window-office-refinement-9-october-2026).

## Optional interior support cleanup

For a saved reconstruction with verified cameras, `scripts/clean_interior.py`
uses only NumPy/SciPy and the existing world alignment/assumed scale. Supply the
full camera/point scene and a world from the same capture/coordinate frame.
The smaller reviewed mask scene is used for later object assembly.

```bash
python scripts/clean_interior.py --scene scenes/room --splats refinement/best.ply --world reviewed-world/world.json --out interior-cleanup --test-every 8
playworld world reviewed-scene --splats interior-cleanup/cleaned.ply --out cleaned-world --object-filter connected --clean-splats --bounds-margin 6
```

`cleaned.ply`, `keep.npy` and `cleanup.json` retain the original property order,
SH coefficients and source hashes. Output must be fresh. The filter rejects
centers farther than `--distance 0.3` assumed meters from sparse points, only
inside the camera footprint (`--margin 0.05`) and `--min-height 0.25` /
`--max-height 2.75`. Exterior/boundary geometry and heights outside that band
are preserved. `--test-every 8` excludes validation-camera centers, matching
the measured trainer split; default zero uses all provided cameras.
This is a heuristic support filter, not free-space measurement or a fix for
camera errors. It can remove real untracked interior surfaces. Review its output.

The GPU refinement tool can apply the same one-time filter directly to checkpoint
tensors before creating fresh optimizers, using training-camera centers only:

```bash
CUDA_HOME=/usr/local/cuda-12.8 OMP_NUM_THREADS=4 "$HOME/.local/share/playworld/refine-gpu/bin/python" scripts/refine_checkpoint.py --scene scenes/room --checkpoint refinement/best.pt --out interior-refinement --interior-world reviewed-world/world.json --steps 500 --eval-every 250 --patch-size 2048 --depth-weight 0.5
```

`--interior-distance`, `--interior-margin`, `--interior-min-height` and
`--interior-max-height` set the matching bounds. Filtering is disabled unless
`--interior-world` is supplied. It is not a persistent training constraint;
Gaussians can subsequently move outside the support threshold. The measured
500-step run took **88.368 s**, reaching **23.1516 dB** on the existing validation
split. A separate `--near-plane` option controls gsplat clipping in raw capture
units (default 0.01); it does not change the browser camera or world geometry.
See the [actual comparison and remaining haze](quality.md#window-office-interior-floater-cleanup-9-october-2026).

## Optional observed-floor footprint and photo window

For haze beyond the camera hull but over observed floor, the support filter
offers an alternative footprint. It remains opt-in; camera mode is the default.
Use the full sparse scene and the same verified y-up world alignment:

```bash
python scripts/clean_interior.py --scene scenes/room --splats refinement/best.ply --world reviewed-world/world.json --out floor-cleanup --footprint floor
```

`--floor-band 0.06` selects reference points near y=0, and `--floor-cell 0.2`
sets horizontal voxel connectivity. The largest component is weighted by
original point count, then its convex hull bounds support cleanup. Existing
distance/inset/height settings apply. Camera centers and `--test-every` do not
affect the floor footprint. Sparse points still come from the provided model;
this is not a strictly isolated training-only reconstruction. All units rely
on the assumed world scale. Missing floors or degenerate footprints fail
explicitly, rather than silently falling back to a different filter.

The GPU tool offers `--interior-footprint floor`, `--interior-floor-band` and
`--interior-floor-cell` together with `--interior-world`. It filters once before
fresh Adam optimizers, not continuously during training. An evaluation-only
check reproduced the CPU/prototype mask count and pre-training metric. The
measured 500-step run used the separately pruned checkpoint and the previous
refinement script; the receipt preserves its actual script hash/version.

For a **manually reviewed photo window**, assemble a raw PLY world first, then
bake and serve the fresh copy:

```bash
playworld world reviewed-scene --splats floor-refinement/best.ply --out floor-world --object-filter connected --clean-splats --bounds-margin 6
python scripts/bake_window_backdrop.py --scene scenes/room --world floor-world --config docs/runs/quality/lounge-window-backdrop-config-20261009.json --out photo-window-world
python -m http.server -d photo-window-world 8000
```

The example configuration is **specific to the window-office trial**. Review
the approximate plane, y/z bounds, photograph names and background strip for
your own capture. Source photographs must be undistorted PINHOLE images and
cover their entire assigned patch. The strip specifies x offsets from
`x = slope*z + intercept`; positive offsets point toward the tested room. It is
not a normal-distance cutoff. The script preserves original files, movable
object PLYs and colliders, hashes its source photographs and refreshes the viewer
in the new world. SPZ packaging also copies and hashes the photo texture.

This is an opaque textured plane, including photographed window frames;
exterior geometry and true parallax are not recovered. Occluding foreground
objects can get baked into it, so the tested fallback covers only y=1.10–2.75 m.
No automatic window detector, inpainting or camera-error correction is claimed.
The normal viewer displays the configuration's disclosure label; recording
receipts identify photo backdrops. For a caption burned into diagnostic media:

```bash
python scripts/encode_gallery.py --frames capture --out docs/runs/quality --name window-trial --title "Window trial" --caption "Window exterior is a photo plane" --gifski path/to/gifski
```

[Measured comparison and remaining artifacts](quality.md#window-office-floor-footprint-and-photo-window-9-october-2026).
