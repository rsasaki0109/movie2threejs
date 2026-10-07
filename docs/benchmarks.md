# Real room benchmark

Measured on 7 October 2026 using an existing Colab NVIDIA L4 (23,034 MiB reported by `nvidia-smi`, driver 580.82.07). No GPU runtime was allocated by the project scripts. Setup, CUDA compilation, earlier failed attempts and already cached model downloads are excluded from the times below.

Input: Eyeful Tower apartment camera 19, a public capture-rig photograph sequence, not a smartphone recording. The workbench section starts at 5.0 s and lasts 4.250 s (51 frames at 12 fps). The pipeline sampled 24 frames at a maximum dimension of 1280 pixels. Source SHA-256: `a0a387a1af7f8483d8c5d13861dc17907054784f090a4c26e8370624b4e4263e`. See [attribution](data-attribution.md) and [derived source metadata](hero-source.json).

## Full reconstruction invocation

[Raw successful report](runs/run-20261007T083553Z.json): **419.079 s** total, below the 15 minute conversion target for this input with cached weights.

| Stage | Measured seconds |
|---|---:|
| frames | 0.949 |
| poses | 143.385 |
| train | 210.861 |
| segment | 55.467 |
| world | 8.360 |

Settings: VGGT with bundle adjustment, one shared camera, initial reprojection tolerance 32 px, tracking batch budget 32,768 frame-point pairs; confidence threshold 1.5; gsplat 7,000 steps. BA converged to 0.567 px final reprojection error ([solver summary](runs/workshop-ba-summary.txt)). Export: 24 cameras, 26,852 points and 875,478 Gaussians. The original object prompts were `chair,box,mug,bottle`; that export contained 14 objects and 2,214 colliders. Its first floor fit selected the workbench and was corrected later.

## Refinements used for the hero

These are separate measured invocations reusing the same poses and trained Gaussians. They are **not a new full-run timing**.

| Invocation | Reused stages | Measured stages | Invocation total |
|---|---|---|---:|
| [More specific object prompts](runs/run-20261007T090338Z.json) | frames, poses, train | segment 44.250 s; world 7.247 s | 51.554 s |
| [Final world assembly](runs/run-20261007T093334Z.json) | frames, poses, train, segment | world 8.408 s | 8.465 s |

Final prompts: `cardboard box,plastic bottle,folding chair`. SAM 3 detected no folding chairs; the final rigid objects are boxes and bottles. Final export: **8 objects, 1,928 static colliders, 8 support patches, 875,243 retained Gaussians**. Object cleanup discarded 235 distant/broad Gaussians. Floor fit: 1,908 sparse inliers; inferred scale 6.298026 m per arbitrary reconstruction unit. This is an assumed metric scale, not a ground-truth measurement.

Corrections: select the supported lower floor instead of the denser workbench; estimate support using compact background splats; carve merged static colliders around bodies; add thin tabletop supports; reject object spill; sample horizontal support colors; fill unseen object interiors with their median captured color. Flat support patches and convex interior fills are approximations, not reconstructed unseen texture.

## Environment and failed attempts

CUDA compiler 13.0. VGGT and gsplat use separate environments with PyTorch 2.9.1+cu130 and NumPy 1.26.4; SAM 3 uses PyTorch 2.10.0+cu130 and NumPy 1.26.4. The source revisions are pinned in [gpu-revisions.env](../scripts/gpu-revisions.env).

The full apartment sequence at the upstream confidence threshold 5.0 exported zero points ([failed report](runs/run-20261007T081011Z.json)). Lowering it to 1.5 completed a no-BA run in 273.476 s ([report](runs/run-20261007T081727Z.json)), but its blur made it unsuitable for the hero. Full-sequence BA first exhausted GPU memory, then lacked enough inliers. The short overlapping workbench section and bounded tracking batches succeeded. These settings are measured for this input; they are not guarantees for other videos.

The shared shell recipe and model APIs ran through the official Colab CLI. The notebook's interactive upload/preview UI was not automated. Smartphone capture, T4 compatibility and true metric accuracy remain unverified.

## Browser recording and exported assets

The local GTX 1660 Ti browser rendered 390 native 1920×1080 frames at a scripted 30 fps, with fixed 60 Hz Rapier steps. Frame capture took **650.214 s**; this is separate from GPU reconstruction time and included concurrent replay QA. The cinematic camera moves independently of the character controller.

[Hero validation](hero-validation.json) records the pushed box's **0.8593 m vertical drop**, 1.2808 m displacement, and inverted final orientation; character movement **1.1000 m**; one thrown ball. Two replays matched at 12 sampled physics states and their first/last PNGs. Every sampled state also matches the complete capture. This does not claim that the ball knocked a mug or bottle off the table.

The final GIF is 960×540 and below 8,000,000 bytes; the MP4 is 1920×1080. Exact measured durations, sizes and SHA-256 values are in [hero-validation.json](hero-validation.json) and [encoding provenance](hero-provenance.json).
