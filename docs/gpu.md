# GPU execution

Use `notebooks/playworld_colab.ipynb` in Colab, or an already provisioned Linux GPU host.
No script creates a VM, Pod, repository, or paid resource. The same setup and run scripts
are used by the notebook and the shell recipe.

The [Colab CLI workflow](colab-cli.md) can transfer and execute the same scripts without
the notebook upload dialogs. Setup, GPU access, trainer help flags and SAM 3 API imports
passed on an existing Colab L4. The full public apartment pipeline completed in 273.476 s
(setup excluded); the first preview has substantial blur and is not the hero asset.

```bash
bash scripts/setup_gpu.sh
python scripts/download_public.py --out scenes/public_source
PLAYWORLD_POSE_CONFIDENCE=1.5 bash scripts/run.sh scenes/public_source/apartment-camera19.mp4
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

Defaults of 24 frames and 7,000 training steps are starting settings, not measured VRAM
or speed guarantees. One no-BA capture completed below the 15 minute target; see
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
