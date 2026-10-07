# GPU execution (not yet validated end to end)

Use `notebooks/playworld_colab.ipynb` in Colab, or an already provisioned Linux GPU host.
No script creates a VM, Pod, repository, or paid resource. The same setup and run scripts
are used by the notebook and the shell recipe.

```bash
bash scripts/setup_gpu.sh
python scripts/download_public.py --out scenes/public_source
bash scripts/run.sh scenes/public_source/apartment-camera19.mp4
```

The setup creates three isolated GPU environments (VGGT, gsplat, SAM 3) plus a lightweight
core environment, pins upstream revisions, checks trainer help flags/imports, and saves
package versions. Linux, Python 3.12, ffmpeg, git, an NVIDIA GPU driver and the CUDA toolkit
(`nvcc`) are required. Set `PLAYWORLD_GPU_ROOT` to relocate environments and repositories.
Setup does not prove model inference works. The local shell syntax and `--check` paths have
been checked; installations, trainer flags at runtime, gated weight access, and GPU inference
still require the Colab run.

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
or speed guarantees. The 15 minute target remains unmeasured. Optional settings:

```bash
PLAYWORLD_NUM_FRAMES=32 PLAYWORLD_TRAIN_STEPS=7000 bash scripts/run.sh room.mp4 scenes/room world
```

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
