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

Final prompts: `cardboard box,plastic bottle,folding chair`. SAM 3 detected no folding chairs; the final rigid objects are boxes and bottles. Original export: **8 objects, 1,928 static colliders, 8 support patches, 875,243 retained Gaussians**. Object cleanup discarded 235 distant/broad Gaussians. Floor fit: 1,908 sparse inliers; inferred scale 6.298026 m per arbitrary reconstruction unit. This is an assumed metric scale, not a ground-truth measurement.

Corrections: select the supported lower floor instead of the denser workbench; estimate support using compact background splats; carve merged static colliders around bodies; add thin tabletop supports; reject object spill; sample horizontal support colors; fill unseen object interiors with their median captured color. Flat support patches and convex interior fills are approximations, not reconstructed unseen texture.

## Environment and failed attempts

CUDA compiler 13.0. VGGT and gsplat use separate environments with PyTorch 2.9.1+cu130 and NumPy 1.26.4; SAM 3 uses PyTorch 2.10.0+cu130 and NumPy 1.26.4. The source revisions are pinned in [gpu-revisions.env](../scripts/gpu-revisions.env).

The full apartment sequence at the upstream confidence threshold 5.0 exported zero points ([failed report](runs/run-20261007T081011Z.json)). Lowering it to 1.5 completed a no-BA run in 273.476 s ([report](runs/run-20261007T081727Z.json)), but its blur made it unsuitable for the hero. Full-sequence BA first exhausted GPU memory, then lacked enough inliers. The short overlapping workbench section and bounded tracking batches succeeded. These settings are measured for this input; they are not guarantees for other videos.

The shared shell recipe and model APIs ran through the official Colab CLI. The notebook's interactive upload/preview UI was not automated. Smartphone capture, T4 compatibility and true metric accuracy remain unverified.

## Browser recording and exported assets

The local GTX 1660 Ti browser rendered 390 native 1920×1080 frames at a scripted 30 fps, with fixed 60 Hz Rapier steps. The current SPZ frame capture took **540.943 s**; this is separate from GPU reconstruction time and included concurrent replay QA. The cinematic camera moves independently of the character controller.

[Hero validation](hero-validation.json) records the pushed box's **0.8593 m vertical drop**, 1.2808 m displacement, and inverted final orientation; character movement **1.0633 m**; one thrown ball. Two replays matched at 12 sampled physics states and their first/last PNGs. Every sampled state also matches the complete capture. This does not claim that the ball knocked a mug or bottle off the table.

The final GIF is 960×540 and below 8,000,000 bytes; the MP4 is 1920×1080. Exact measured durations, sizes and SHA-256 values are in [hero-validation.json](hero-validation.json) and [encoding provenance](hero-provenance.json).

## Additional gallery captures

The original four additional public camera-19 sequences ran on the same existing NVIDIA L4, with cached weights, 16 frames at 1024 pixels, 4,500 gsplat steps, bundle adjustment, a shared camera and an 8,192 frame-point tracking budget. These are distinct real spaces, not different views of the apartment workbench.

Initial attempts exhausted memory while another job used the shared GPU. No other process or runtime was stopped. Successful stages were reused after free memory returned. The historical table lists measured successful stages from the preserved reports; it does **not** describe a single uninterrupted wall-clock conversion. Setup, failed attempts, queueing and browser recording are separate.

| Capture | Frames s | Poses s | Train s | Segment s | First world s | Final objects | Final colliders |
|---|---:|---:|---:|---:|---:|---:|---:|
| [Meeting room](runs/gallery/meeting-room/run-resume-segment.json) | 0.652 | 166.235 | 120.855 | 94.073 | 1.660 | 1 | 1,139 |
| [Kitchen/cafe](runs/gallery/cafe/run-resume-poses.json) | 0.796 | 239.557 | 85.557 | 52.773 | 2.982 | 6 | 2,520 |
| [Window office](runs/gallery/lounge/run-resume-poses.json) | 0.911 | 147.269 | 80.370 | 35.394 | 2.716 | 2 | 4,169 |
| [Furnished room](runs/gallery/furnished-room/run-resume-poses.json) | 0.787 | 120.001 | 67.721 | 35.912 | 1.206 | 4 | 1,471 |

Meeting-room poses and training succeeded in [earlier](runs/gallery/meeting-room/run.json) [invocations](runs/gallery/meeting-room/run-resume-train.json). Raw reports retain the wrapper-import failure and out-of-memory failures. The cafe, window-office and furnished-room successful invocations reused the already extracted frames and took 380.923 s, 265.825 s and 224.899 s respectively.

Browser review rejected unstable/fragmentary SAM instances. The meeting room was rebuilt with chair instance 9 only ([1.316 s assembly](runs/gallery/meeting-room/reviewed-assembly.json)); the cafe keeps six reviewed instances ([3.386 s assembly](runs/gallery/cafe/reviewed-assembly.json)). Other captured furniture stays static. This selection is manual, not a claim of automatic perfect segmentation.

The window office initially produced enormous collision boxes: distant sparse points expanded voxel bounds and forced excessive coarsening. Rebuilding with a 6 m margin around the aligned camera path excluded 12,439 distant Gaussians and clipped collision/support inputs ([2.820 s assembly](runs/gallery/lounge/bounded-assembly.json)). The default pipeline does not crop unless `--bounds-margin` is set. The synthetic-room regression checks that distant outliers cannot enlarge the bounded room or remove its chair, crate and mug.

Local native GTX 1660 Ti capture produced 120 frames per additional room, at 1280×720 and scripted 15 fps. Original capture times were 86.168 s (meeting room), 53.871 s (cafe), 49.178 s (window office) and 51.267 s (furnished room); this is separate from reconstruction. Each preview lasts 8 s and uses fixed 60 Hz physics. [Gallery checks](gallery-validation.json) confirm a hit on the designated object, visible movement, walking, one thrown ball and identical replay states in every room. These previews show approximate masks, blur and simple colored unseen interiors; they are not a photorealism benchmark.


## Quality comparison and current exports

These refinements reuse the existing L4 and cached inputs; they are separate from the original full-run times above. Shorter runs now scale gsplat's complete densification/reset/settling schedule. Experimental pose optimization and volume regularization were tested for 15,000 steps using the same sixteen VGGT views and masks.

| Capture | Train seconds | Held-out PSNR before / after | Decision |
|---|---:|---:|---|
| [Meeting room](runs/quality/meeting-room-experimental-train.json) | 417.518 | 20.2303 / 19.7532 | Reject retraining |
| [Kitchen/cafe](runs/quality/cafe-experimental-train.json) | 558.734 | 17.5366 / 18.4538 | Adopt; world assembly 7.150 s |
| [Furnished room](runs/quality/furnished-room-experimental-train.json) | 464.756 | 19.8278 / 18.4856 | Reject retraining |

The adopted cafe has 6 objects and 2,197 colliders. Its 1,063,521 retained Gaussians occupy 20,510,053 bytes of SPZ. Improved held-out error does not establish sharp arbitrary-view rendering; blur remains.

The workbench, window office and furnished room retain their original trained geometry and physics, with conservative background-volume removal and SH3-preserving SPZ export. Export/cleanup CPU times were 6.966 s, 2.334 s and 0.687 s respectively ([reports](runs/quality/)). Background volumes removed: 878, 4,733 and 1,874. Current workbench: 874,365 Gaussians, 8 objects, 1,928 colliders. These counts supersede the original export counts; they are not new complete pipeline runs.

Blanket opacity deletion and filtering against sparse reference planes were rejected after rendered comparison because they opened holes in walls. The optional training preset remains experimental. See [quality notes](quality.md).


## Calibrated meeting-room showcase

The adopted meeting room uses **192 undistorted photographs** from Eyeful Tower office1b cameras **19,16,22**, with **provided COLMAP camera calibration and sparse point initialization**. VGGT is bypassed. This is a visual showcase, not a full video-to-world or smartphone benchmark. Training uses 30,000 default gsplat steps. SAM masks use sixteen camera-19 views.

| Measured stage on existing L4 | Seconds |
|---|---:|
| Download and prepare 192 photographs plus calibration | 76.258 |
| gsplat training | 704.616 |
| SAM 3 seed-frame-5 trial (rejected masks) | 29.755 |
| Initial world with rejected masks | 6.027 |
| Reviewed world, reusing earlier seed-frame-0 masks | 5.410 |

The reused seed-frame-0 SAM invocation took 31.772 s in the earlier 64-photo experiment. Do not add the rejected and reused paths into a claimed single uninterrupted conversion. Setup, export, queueing and recording are separate. [Full run and selection](runs/quality/calibrated-multi-192.json); [earlier masks and training](runs/quality/calibrated-64.json).

Reviewed instance: chair 1 only. Other chairs remain static. Final export: **1 movable object, 1,936 static colliders, 1 support patch, 617,815 Gaussians**, 13,306,177 bytes SPZ. Assumed scale: 1.323943 m per source unit, using the same 1.5 m eye-height assumption. Held-out PSNR 26.9179, SSIM 0.900942, LPIPS 0.246921 on 24 held-out views. These views differ from earlier trials; the scores are not a controlled comparison with video/VGGT results.

Browser comparison found clearer walls, table and floor at the selected photographed viewpoints, with persistent close mesh-chair streaks and incomplete exposed surfaces after the push. Blanket opacity and sparse-plane filtering were rejected. No inpainting or photographic completion was introduced.

Current native GTX 1660 Ti recording times: workbench 540.943 s (390 frames,
1920×1080), meeting room 91.357 s, cafe 85.834 s, window office 74.693 s,
furnished room 77.322 s (120 frames each, 1280×720). [Recording receipts](runs/quality/browser-captures.json)
bind timings to exact world and shot hashes. [Final gallery checks](gallery-validation.json)
verify all five current exports: identical physics replays, featured pushes, a thrown
ball, scripted walking and actual keyboard walking. The meeting-room chair moved
0.9752 m after the push; its initial settling was 0.0592 m in assumed units.
