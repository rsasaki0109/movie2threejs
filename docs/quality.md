# Reconstruction quality

The room previews have visible blur and incomplete coverage. Gaussian pruning and a
longer training run do not reconstruct surfaces that were never observed.

## Reproducible corrections

Short training now scales the entire gsplat schedule with `--steps-scaler`, including
densification, opacity resets and the final settling period. Previously, reducing only
`max_steps` stopped training partway through the default 30,000-step strategy.

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

Only reviewed MIT-licensed Eyeful Tower scenes are accepted. Keep the dataset's
[MIT notice](EYEFULTOWER-LICENSE.txt) with redistributed outputs. Dataset attribution
does not change model-weight licenses or demonstrate smartphone capture quality.
