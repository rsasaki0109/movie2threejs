# Reconstruction quality

The room previews have visible blur and incomplete coverage. Gaussian pruning and a
longer training run do not reconstruct surfaces that were never observed.

## Reproducible corrections

Short training now scales the entire gsplat schedule with `--steps-scaler`, including
densification, opacity resets and the final settling period. Previously, reducing only
`max_steps` stopped training partway through the default 30,000-step strategy.

The CLI now schedules weight checkpoints and PLY exports at six milestones across
that scaled schedule. A 30,000-step run saves every 5,000 steps. This replaces the
final-only setting that lost the interrupted calibrated workbench at step 26,995.
Regression tests cover interruption during both 7,000- and 30,000-step command
schedules. The 8 October L4 trial actually saved all six checkpoints and PLYs at
5,000-step intervals. All six weight files were reloaded on CPU with finite tensors
([validation](runs/quality/workbench-checkpoint-validation.json)). The selected
5,000-step and final 30,000-step weights, photograph inputs, calibration and masks
were downloaded and checked locally before finishing the GPU work.
Upstream checkpoints preserve Gaussian weights and the saved step, but do not
contain the training optimizers: they are not exact training-resume snapshots.

`playworld world --clean-splats` conservatively rejects thick background volumes in the
aligned, assumed metric frame. Thin walls and low-opacity thin detail are retained.
The synthetic regression verifies that adding a diffuse background blob removes that
blob without changing collision geometry. Scale remains an estimate.

Blanket opacity pruning was rejected after browser comparison: it made dark holes in
the meeting-room lighting. Sparse-reference plane filtering also removed valid wall
coverage and was rejected. The optional `--quality` training preset requires
comparison rather than automatic adoption; camera optimization and volume regularizers
made some measured held-out views worse.

## Browser color and compression

`scripts/build_demo.py --format spz` preserves higher-order spherical harmonics through
SH3 using quantized SPZ v3. The older `.splat` export retains only DC color. This restores
view-dependent color, but cannot fix incorrect geometry. The encoder keeps the capture
axes because `world.align` already converts them to the viewer's world coordinates.
An extra coordinate flip would mirror or displace the reconstruction.

Optional encoder setup, in the isolated Linux/Colab core environment:

```bash
bash scripts/setup_spz.sh
```

## Calibrated public showcase

The public-data preparation script deliberately uses the dataset's COLMAP camera
calibration **and sparse point initialization**, plus its undistorted JPEG photographs.
It does not invoke VGGT or estimate camera poses from a video. This distinguishes a
visual showcase of the playable-world builder from a test of automatic video conversion.

```bash
pip install -e '.[showcase]'
python scripts/prepare_calibrated_public.py office1b --out scenes/calibrated-meeting --frames 64 --camera 19,16,22
```

With the isolated environments from `scripts/setup_gpu.sh`, the corresponding stages
are explicit. Adjust interpreter paths when using another GPU root:

```bash
.gpu/envs/core/bin/playworld train scenes/calibrated-meeting --gsplat-dir .gpu/repos/gsplat --python .gpu/envs/gsplat/bin/python --steps 30000
.gpu/envs/core/bin/playworld segment scenes/calibrated-meeting/segment-scene --python .gpu/envs/sam3/bin/python --prompts chair --seed-frame 0
.gpu/envs/core/bin/playworld world scenes/calibrated-meeting/segment-scene --splats scenes/calibrated-meeting/gs/ply/point_cloud_29999.ply --out calibrated_world --bounds-margin 6 --clean-splats
```

Use the PLY filename actually exported by the pinned trainer. This recipe needs a
GPU with sufficient free memory; sharing, queueing and setup are outside stage timings.

The committed meeting room retains manually reviewed instance 1 from the seed-frame-0
masks. A later seed-frame-5 trial yielded chair fragments and a floor false positive;
those masks were rejected. Running the commands above produces unreviewed masks:
inspect them and keep the desired instance before packaging a showcase.

The adopted 192-photo / 30,000-step run reached held-out PSNR 26.9179, SSIM 0.90094
and LPIPS 0.24692. The earlier 64-photo / 15,000-step default trial reached 23.1405,
0.87319 and 0.30527. These use different training and validation views, so this is
not a controlled improvement attributed solely to one setting. Rendered review found
clearer walls, tabletop and floor from the chosen capture viewpoints. Close mesh-chair
backs still streak; moving the chair exposes incomplete surfaces. The room is not
artifact-free and this does not establish arbitrary-view photorealism.

[Exact calibrated run and manual selection](runs/quality/calibrated-multi-192.json).

The preparation manifest records each photograph and reconstruction file's URL, byte
count and SHA-256. Intrinsics are resized to the exact downloaded image dimensions;
camera rotations, translations and point coordinates retain their shared source frame.
SAM 3 can run on the generated sixteen-view `segment-scene`, in that same frame,
while gsplat trains on all selected photographs.

For a selected part of a longer capture, `--start-index` and `--stop-index` select
an ordinal interval independently in each camera's sorted photographs. The stop
index is exclusive; `--frames` is a maximum per camera. This apartment interval
downloads 52 photographs per camera, 156 total, rather than training on unrelated
parts of the full apartment sequence:

```bash
python scripts/prepare_calibrated_public.py apartment --out scenes/calibrated-workbench --frames 64 --camera 19,16,22 --start-index 60 --stop-index 112
```

The manifest retains the interval and exact photographs. Ordinals refer to sorted
capture photographs, not seconds in an arbitrary input video.

`scripts/run_calibrated.py` runs preparation, training, segmentation and assembly
with the same isolated GPU environments, and writes stage logs and a timing report.
It checks SAM 3 access before preparing data and requires fresh output directories.
For the selected workbench interval, the prepared command is:

```bash
python scripts/run_calibrated.py apartment --scene scenes/calibrated-workbench-run --out calibrated_workbench_world --camera 19,16,22 --frames 64 --start-index 60 --stop-index 112 --steps 30000 --seed-frame 5
```

Install `.[showcase]` in the core environment for calibrated-image preparation.
The underlying preparation, training, segmentation and assembly commands completed
on an existing L4 on 8 October. The complete portable wrapper has not been invoked
end to end; the measured run used an equivalent orchestration script. Review masks
before publishing outputs. An initial SPZ export used the wrong Python environment;
repeating CPU assembly/export with `.gpu/envs/core/bin/python` resolved it without
another training run.

The current README workbench keeps **`point_cloud_4999.ply` from that 30,000-step
trial**, rather than the final PLY. This is not equivalent to a separately scaled
5,000-step run. The same camera path was rendered with reviewed label-1 box masks
at 5,000, 10,000 and 30,000 steps. The early export looked smoother; the final export
had finer speckling on the workbench and floor. The final held-out PSNR of 30.6495
is a metric for the rejected 30,000-step weights, not for the adopted hero, and the
held-out views differ from the earlier VGGT experiment.

The adopted export has **1,110,159 Gaussians, one movable box, 2,834 static
colliders and one support patch**, with 20,089,789 bytes of SPZ. Bounds cleanup
removed 27,647 Gaussians, diffuse cleanup 3,823, and object-spill cleanup five.
Only the reviewed box moves; other SAM detections remain static. Vertex colors
sampled from the observed support surface reduce the dark hole left after the box
moves. This remains a flat approximation; the unknown underside uses a colored
convex interior. Floor blur, mask-edge fragments and incomplete views remain.

The first proposed player position overlapped low collision boxes and could not
walk. The committed profile moves its XZ start by approximately 0.72 m to a clear
location; camera keyframes still follow the capture viewpoints. Real keyboard and
touch-emulated walking, a falling/inverting box and repeated fixed-step physics were
checked. The cinematic camera is independent of the capsule controller. The
[measured trial and selection](runs/quality/calibrated-workbench-156-selected-5000.json)
and [hero checks](hero-validation.json) separate reconstruction from browser results.

Only reviewed MIT-licensed Eyeful Tower scenes are accepted. Keep the dataset's
[MIT notice](EYEFULTOWER-LICENSE.txt) with redistributed outputs. Dataset attribution
does not change model-weight licenses or demonstrate smartphone capture quality.
