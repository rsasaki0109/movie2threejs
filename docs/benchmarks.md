# Real room benchmark

**Current hero:** the [8 October calibrated workbench](#current-calibrated-workbench-8-october-2026)
uses 156 provided photographs and the 5,000-step checkpoint from a 30,000-step trial.
The original video/VGGT timings below are preserved as historical measurements;
they do not describe the current hero or its conversion time.

The historical video/VGGT runs below were measured on 7 October 2026 using an existing Colab NVIDIA L4 (23,034 MiB reported by `nvidia-smi`, driver 580.82.07). No GPU runtime was allocated by the project scripts. Setup, CUDA compilation, earlier failed attempts and already cached model downloads are excluded from those times.

Input: Eyeful Tower apartment camera 19, a public capture-rig photograph sequence, not a smartphone recording. The workbench section starts at 5.0 s and lasts 4.250 s (51 frames at 12 fps). The pipeline sampled 24 frames at a maximum dimension of 1280 pixels. Source SHA-256: `a0a387a1af7f8483d8c5d13861dc17907054784f090a4c26e8370624b4e4263e`. See [attribution](data-attribution.md) and [derived source metadata](hero-source.json).

## Controlled camera comparison: 8 October 2026

Two static reconstructions used the same **24 undistorted photographs, provided
intrinsics and 100,000 initial points/colors** on a dedicated NVIDIA L4
(23,034 MiB, driver 580.82.07). Only the camera extrinsics differed: the saved
7 October VGGT estimates aligned into the reference frame, or the dataset's
provided cameras. Both used 7,000 gsplat steps and identical CLI options.
This is a camera-input quality experiment, not a video pipeline, segmentation
benchmark or replacement for the current 156-photo hero.

| Measurement | Estimated cameras | Provided cameras |
|---|---:|---:|
| Train, seconds | 327.072 | 378.877 |
| World export, seconds | 5.453 | 6.416 |
| Gaussians | 1,322,057 | 1,695,831 |
| Movable objects | 0 | 0 |
| Static colliders | 3,187 | 4,061 |
| Mean PSNR from common provided held-out cameras, dB | 15.18 | 28.49 |
| Mean SSIM from common provided held-out cameras | 0.657 | 0.874 |

Training, export and backup preparation for both conditions took **757.265 s**,
excluding installation, an initial backend-setting failure and local recovery.
The trainer's scene extent is derived from camera centers and also differs
between conditions; this does not isolate rotation from positional errors or
their effect on training rates and thresholds.

Both final checkpoints and world ZIPs were recovered and checked before the
task-owned L4 was terminated. Elapsed allocation/validation time was **29 min
13.2 s**; consumption at the observed 1.54 units/hour is **estimated** at
**0.749972 units**, rather than individually measured billing. See [all three
same-camera comparisons, raw metrics and artifact validation](pose-comparison.md).

## Interactive notebook run: 8 October 2026

The imported Drive notebook completed its public video/VGGT recipe on a dedicated
Colab **NVIDIA L4 (23,034 MiB, driver 580.82.07)**. The saved notebook output
contains the [successful final report](colab-ui-run-20261008.json).
Conversion took **745.670 s (12 min 25.7 s)**, including initial model downloads.
Environment installation took **601.612 s** and source download **7.728 s**,
both separate from conversion. This meets the 15-minute conversion target for
this particular public input; it is not the current curated hero's timing.

| Stage | Measured seconds |
|---|---:|
| frames | 0.918 |
| poses | 226.935 |
| train | 438.415 |
| segment | 63.142 |
| world | 16.204 |

Settings: 24 frames, maximum dimension 1280, VGGT bundle adjustment with a shared
camera, confidence threshold 1.5, initial reprojection tolerance 32 px and
7,000 gsplat steps. Export: **8 movable objects, 2,962 static colliders,
8 support patches and 1,982,706 Gaussians**. Sparse reconstruction contained
28,440 points. Eye height was assumed to be 1.5 m; inferred scale was
6.292355 m per reconstruction unit, without ground-truth metric validation.

The public photograph sequence was re-encoded on this runtime. Its input MP4
was 9,198,472 bytes, SHA-256
`6529e7d13119873a71aeeae2499bd778dc35bfbd57d36b9c838ee2b24f31e79a`.
This is different from the historical MP4 above, so the two runs are not a
controlled speed or quality comparison. No smartphone recording was tested.

The dedicated runtime was stopped within the approved 30-minute limit,
about 18 seconds after conversion finished. The saved output and an intermediate
step-4,665 checkpoint were preserved; the final world and result ZIP were not
downloaded. Browser preview, result download and visual quality of this export
remain unverified. See [interactive verification details](colab-ui.md).

## Historical full reconstruction invocation

[Raw successful report](runs/run-20261007T083553Z.json): **419.079 s** total, below the 15 minute conversion target for this input with cached weights.

| Stage | Measured seconds |
|---|---:|
| frames | 0.949 |
| poses | 143.385 |
| train | 210.861 |
| segment | 55.467 |
| world | 8.360 |

Settings: VGGT with bundle adjustment, one shared camera, initial reprojection tolerance 32 px, tracking batch budget 32,768 frame-point pairs; confidence threshold 1.5; gsplat 7,000 steps. BA converged to 0.567 px final reprojection error ([solver summary](runs/workshop-ba-summary.txt)). Export: 24 cameras, 26,852 points and 875,478 Gaussians. The original object prompts were `chair,box,mug,bottle`; that export contained 14 objects and 2,214 colliders. Its first floor fit selected the workbench and was corrected later.

## Refinements used for the historical video/VGGT hero

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

These historical runs used the official Colab CLI. The later interactive
notebook run is recorded above; its browser preview and result download remain
unverified. Smartphone capture, T4 compatibility and true metric accuracy remain
unverified.

## Historical video/VGGT browser recording

The local GTX 1660 Ti browser rendered 390 native 1920×1080 frames at a scripted 30 fps, with fixed 60 Hz Rapier steps. The previous VGGT SPZ capture took **540.943 s**; this is separate from GPU reconstruction time and included concurrent replay QA. The cinematic camera moves independently of the character controller.

[Historical hero validation](runs/quality/workbench-vggt-hero-validation.json) records the pushed box's **0.8593 m vertical drop**, 1.2808 m displacement, and inverted final orientation; character movement **1.0633 m**; one thrown ball. Two replays matched at 12 sampled physics states and their first/last PNGs. Every sampled state also matched the complete capture. This does not claim that the ball knocked a mug or bottle off the table.

That GIF was 960×540 and below 8,000,000 bytes; the MP4 was 1920×1080. Historical sizes and hashes are preserved in the validation linked above. Current export measurements are in [hero-validation.json](hero-validation.json) and [encoding provenance](hero-provenance.json).

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

Browser review rejected unstable/fragmentary SAM instances. The original meeting room was rebuilt with chair instance 9 only ([1.316 s assembly](runs/gallery/meeting-room/reviewed-assembly.json)); the original cafe kept six reviewed instances ([3.386 s assembly](runs/gallery/cafe/reviewed-assembly.json)). Both were subsequently replaced by calibrated showcases. Other captured furniture stays static. This selection is manual, not a claim of automatic perfect segmentation.

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

The earlier workbench, window office and furnished room retained their original trained geometry and physics, with conservative background-volume removal and SH3-preserving SPZ export. Export/cleanup CPU times were 6.966 s, 2.334 s and 0.687 s respectively ([reports](runs/quality/)). Background volumes removed: 878, 4,733 and 1,874. The historical workbench had 874,365 Gaussians, 8 objects and 1,928 colliders; it has since been replaced by the calibrated workbench below. Window office and furnished room still use those exports.

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

Earlier native GTX 1660 Ti recording times: workbench 540.943 s (390 frames,
1920×1080), meeting room 91.357 s, cafe 85.834 s, window office 74.693 s,
furnished room 77.322 s (120 frames each, 1280×720). [Recording receipts](runs/quality/browser-captures.json)
bind timings to exact world and shot hashes. [Final gallery checks](gallery-validation.json)
verify all five current exports: identical physics replays, featured pushes, a thrown
ball, scripted walking and actual keyboard walking. The meeting-room chair moved
0.9752 m after the push; its initial settling was 0.0592 m in assumed units.

## Current calibrated workbench (8 October 2026)

GPU: **existing shared NVIDIA L4, 23,034 MiB**. The attempt to create a new runtime
was refused by Colab's assignment limit; the successful run attached to an existing
runtime. No new paid resource was created. Other jobs were not terminated. gsplat
used PyTorch 2.9.1+cu130 and SAM 3 used 2.10.0+cu130 in separate environments.
The trial limited its Torch allocator to 45% of the device.

Input: **156 undistorted Eyeful Tower apartment photographs**, cameras **19,16,22**,
52 each, ordinal interval **60:112** with exclusive stop. Dataset-provided COLMAP
calibration and 100,000 sparse initialization points replace VGGT. SAM 3 uses
16 camera-19 views, seed frame 5 and prompts `cardboard box,plastic bottle`.
This is a calibrated public showcase, not a smartphone/video-pose benchmark.

| Measured stage | Seconds |
|---|---:|
| Isolated GPU environment setup | 626.147 |
| SPZ dependency setup | 43.531 |
| Download and prepare photographs/calibration | 110.825 |
| Full default 30,000-step gsplat trial | 2,363.596 |
| SAM 3 inference with cached downloaded weights | 39.922 |
| Initial unreviewed 30,000-step world | 17.508 |
| Selected 5,000-step world with reviewed masks | 9.524 |
| Selected SPZ export in core environment | 3.483 |

SAM 3 weight prefetch separately took **16.460 s** (3,450,062,241 bytes).
Setup, weight transfer, rejected variants, download/backup, queueing and browser
capture are outside the reconstruction stage times. The complete training trial
took **39 min 23.6 s** and did not meet the 15-minute target. The trainer's cumulative
timer reached its 5,000-step save at **118.724 s**; this is an intermediate timer,
not a measured complete 5,000-step conversion. A scaled 5,000-step run is a different
densification schedule.

All six actual checkpoint and PLY saves at 5k/10k/15k/20k/25k/30k were produced.
[CPU reload validation](runs/quality/workbench-checkpoint-validation.json) checked
finite tensors in all six weight files. Selected 5k and final 30k weights, inputs,
calibration, masks and configuration were downloaded and SHA-256 checked locally
([backup verification](runs/quality/workbench-local-backup-verification.json)).
The checkpoints store weights and step, **not optimizer state**.

The initial export failed because the orchestration script used native `python3`
without the isolated SPZ dependency. Repeating CPU world/export in the core
environment fixed it; training and segmentation were reused.

Manual browser comparison of the same camera path selected **5,000-step weights**.
10k and 30k exports were larger (42,088,483 and 64,157,149 bytes SPZ) and showed more
floor/tabletop streaks or speckling at these viewpoints. The rejected final 30k
weights scored PSNR **30.6495**, SSIM **0.903577**, LPIPS **0.157682** on their held-out
views. These are not adopted-hero metrics and not a controlled comparison with
the earlier VGGT data. [Trial and selection](runs/quality/calibrated-workbench-156-selected-5000.json).

Selected export before local box refinement: **1 reviewed box, 2,834 static
colliders, 1 support patch, 1,110,159 retained Gaussians, 20,089,789 SPZ bytes**. Bounds pruning removed 27,647
Gaussians, diffuse pruning 3,823, object-spill pruning five. Floor fit: 21,797 inliers;
assumed scale **1.323942 m per source unit**, still based on 1.5 m camera height.
Other detections remain static. The player start was moved to a clear position
after a first start overlapped low collision boxes. This does not guarantee every
part of the room has correct collision geometry.

The current [hero validation](hero-validation.json) records the complete 390-frame,
1920×1080 capture, actual box fall/tip, character movement, one ball, matching
12-state replay samples and matching first/last PNGs. [Browser capture receipts](runs/quality/browser-captures.json)
separate recording time from reconstruction. Keyboard walking, pointer lock,
touch-emulated walking/look/throw and all five gallery rooms were checked locally.
Later checks also passed on the [public GitHub Pages deployment](live-demo.md#github-pages).
Blur in unseen floor regions and colored
unobserved box faces remain visible.

Pre-refinement workbench browser capture: **559.327 s**, native GTX 1660 Ti,
390 frames at 1920×1080, including concurrent sampled replay QA. The pushed box
dropped **1.0236 m** and finished inverted; character XZ movement was **0.4379 m**.
One ball was created. GIF size: **7,811,010 bytes**; duration **13.1 s**.
The 1920×1080 MP4 is **9,207,385 bytes**. These are measured browser/export
results, separate from training.

## Current CPU box refinement and re-export

The selected weights and local reviewed masks/calibration were reused; no new
Colab session or GPU training was used. The final CPU script replay took **5.744 s**,
excluding dependency setup, checkpoint-to-PLY recovery and SPZ packaging. Its world
JSON and object PLY matched the adopted candidate byte for byte. The original
background SPZ, collision body, mass, support patch and 2,834 colliders are unchanged.
The object keeps 8,551 of 9,258 splats (707 removed); the final world has
**1,109,452 Gaussians and 20,079,956 SPZ bytes**. The box uses a small approximate
horizontal bottom rather than its whole convex collision hull for interior fill.

The new native GTX 1660 Ti recording took **584.769 s**, including concurrent
replay/gallery QA: 390 frames, 1920×1080, scripted 30 fps, fixed 60 Hz physics.
All **390 physical states equal the previous hero**, with two matching 12-state
replays and matching first/last PNGs. The box still drops **1.0236 m** and finishes
inverted; player XZ movement is **0.4379 m**, and one ball is thrown.

GIF: **7,798,040 bytes**, 960×540, looping **13.1 s**. MP4: **9,225,667 bytes**,
1920×1080, **13.066667 s**. [Refinement and rejected trials](runs/quality/workbench-box-polish.json),
[hero validation](hero-validation.json), [before/after video](demos/workbench-box-comparison.mp4).
These are browser/export measurements, not additional GPU conversion times.
The hidden tabletop and keyboard remain approximate; blur is not fully resolved.

## Colab Secrets and setup check, 8 October 2026

A separate updated-notebook check fetched source commit
`264f62f02b46b7dcf7fab53eed85c89163fb45c6`, verified gated SAM 3 access through
Colab Secrets, and completed isolated environment/API setup in **619.101 s**.
GPU: **NVIDIA L4, 22,563 MiB reported by PyTorch**, native bf16 support.
No input, pose estimation, training, segmentation or world assembly ran in this
attempt, so it has no new object/collider counts or inference timings.

The browser connection was lost during setup. Its log and `job.json` were
downloaded through the CLI and the dedicated L4 was terminated within the
approved limits. The unrelated T4 was left running. This verifies Secrets and
installation, not a complete updated GPU notebook run or final-checkpoint
backup. [Receipt](colab-gpu-preflight-20261008.json),
[details and remaining checks](colab-ui.md#updated-l4-secrets-and-setup-check).

## Updated Colab GPU reconstruction, 8 October 2026

The automatic-source notebook ran setup and the public video/VGGT pipeline
together at source revision `264f62f02b46b7dcf7fab53eed85c89163fb45c6`.
GPU: **NVIDIA L4**, 23,034 MiB reported by nvidia-smi and 22,563 MiB by
PyTorch; native bf16 support. SAM 3 access used Colab Secrets.
The MIT-licensed Eyeful Tower capture-rig photograph sequence was encoded as
`apartment-workshop.mp4`; it is not a phone recording. The input SHA-256,
24-frame/7,000-step settings and raw reports are in the
[execution receipt](colab-gpu-reconstruction-20261008.json).

| Stage | Measured seconds |
| --- | ---: |
| Isolated environment setup | 580.165 |
| Public source download | 5.348 |
| Frame extraction | 0.942 |
| VGGT poses and bundle adjustment | 242.057 |
| gsplat training, 7,000 steps | 444.655 |
| SAM 3 segmentation | 62.130 |
| World assembly | 13.800 |
| Core pipeline total | 763.642 |
| Enclosing notebook pipeline cell | 767.584 |

Core timing includes initial model downloads and excludes setup/source download.
World report: **8 objects, 2,957 static colliders, 7 support patches and
1,868,515 exported Gaussians**. This scene has not received visual review.

The new ZIP and final checkpoint downloads were incomplete when the L4 stopped
within its approved 40-minute window. Neither archive integrity nor final
checkpoint CPU loading was verified. This is a successful reconstruction
measurement, not a complete recoverable scene or end-to-end first-run validation.
[Export failure and remaining checks](colab-ui.md#updated-gpu-reconstruction-and-incomplete-export).

## Verified GPU backups, 8 October 2026

A dedicated **NVIDIA L4** ran the notebook's first six code cells at source
`41d01a505362a95bfa3144a05e70ca5f646b1eb3` through native Windows networking.
Input: the MIT-licensed Eyeful Tower apartment-workshop capture-rig photograph
sequence encoded as video, **24 frames and 7,000 training steps**. This is not a
phone capture. SAM 3 used a one-use approved-account token transfer; Secrets had
been checked separately. The input hash and complete raw reports are in the
[verified backup receipt](colab-gpu-backup-20261008.json).

| Stage | Measured seconds |
| --- | ---: |
| Isolated environment setup | 586.500 |
| Public source download | 7.763 |
| Frame extraction | 0.968 |
| VGGT poses and bundle adjustment | 257.457 |
| gsplat training, 7,000 steps | 446.000 |
| SAM 3 segmentation | 62.880 |
| World assembly | 14.910 |
| Core pipeline total | 782.271 |
| Enclosing notebook pipeline execution | 786.504 |

Core timing includes initial model downloads and excludes environment setup and
source download. Export: **8 objects, 3,060 static colliders,
6 support patches and 2,047,651 Gaussians**. This was the default
video/VGGT recipe, without diffuse/bounds cleanup or calibrated dataset cameras.

The **483,453,685-byte final checkpoint** and **450,756,877-byte world ZIP**
were fully recovered, with matching SHA-256, before runtime termination. ZIP CRC,
world references and viewer/source identity passed. CPU weight loading verified
step 6,999 and all finite tensors for **2,048,520 training Gaussians**; optimizer
state is absent. The 32-worker index watcher completed; an independent concurrent
32-worker ZIP copy also completed with the same hash. No manual chunk retry was
needed. These transfer observations do not guarantee the same rate on another run.

The runtime stopped after **44 min 18.1 s**, inside the original 45-minute window.
A five-minute extension was approved but not needed. Estimated task consumption
was **1.137059 units** at the initially inferred 1.54 units/hour; individual billing
was not measured. After shutdown, the local exported-world browser passed W/A/S/D
walking, drag look, ball creation, pause/resume/reset and pointer lock, with no
page errors. Significant blur and streaks remain; this result was not adopted
for the hero/gallery. The interactive Colab preview/download cells and phone-video
upload were not verified in this GPU attempt. The unchanged preview/download cells
subsequently passed separately on Colab CPU using this actual world: **9.479 s**
to restore, **450,612,560 bytes** downloaded through the browser, ZIP CRC and all
**13 world files** matching the GPU archive. Browser rendering, drag look, visible
ball throwing and pause passed. CPU allocation lasted **14 min 41.3 s**, with
**0.0195845 units** estimated at the inferred 0.08 units/hour; individual billing
was not measured. No new reconstruction was run. A same-session interactive GPU
first run and phone-video upload remain unverified.
[Preview and verification limits](colab-ui.md#actual-gpu-world-restored-in-colab).

## Calibrated kitchen: 9 October 2026

The adopted kitchen/cafe uses **144 undistorted Eyeful Tower photographs** from
cameras **19,16,22**, ordinal interval **168:240** independently per camera,
provided Metashape COLMAP calibration and **100,000 sparse initialization points**.
SAM 3 used sixteen of camera 19's photographs in the same coordinate frame,
with prompt `chair` and seed frame 4. No VGGT or smartphone video was used.
The final 7,000-step model contains **1,210,857 Gaussians**.

Source revision: `634a5e249f95a9a7b8d40a6426b9324095bb5eed`.
Dedicated **NVIDIA L4, 23,034 MiB**, driver **580.82.07**.
Dates here use JST; the UTC run and termination timestamps are in the
[complete receipt](runs/quality/cafe-calibrated-20261009.json).

| Stage | Measured seconds |
| --- | ---: |
| Isolated environment setup, including CUDA compilation | 602.029 |
| gsplat training, 7,000 steps | 290.618 |
| SAM 3 segmentation | 48.651 |
| Raw world assembly | 9.040 |
| Train + segment + raw world | 348.309 |
| Enclosing run, including input extraction and backup preparation | 380.015 |
| Local CPU rebuild after selecting chair 1 | 8.470 |
| Native 1920×1080 browser recording, 120 frames | 140.984 |

The core stages exclude setup, locally prepared input downloads, result transfer,
CPU review and browser recording. The raw export has **4 movable objects and
4,910 colliders**. Manual mask/shape review retains **one foreground chair**;
thin or incomplete detections remain static. The adopted world has **1,200,800
Gaussians, 4,916 colliders and one approximate floor patch**, packed as
**23,183,263 SPZ bytes** with SH3 retained. Its upper interior is an approximate
colored hull clipped at assumed height **0.38 m**; physics geometry is unchanged.

Trainer held-out metrics were **26.874 dB PSNR, 0.9097 SSIM and 0.2399 LPIPS**.
The photographs, input count, poses and training schedule differ from the older
published cafe, so these are not a controlled before/after comparison.
Floor and vegetation blur remain visible.

The **285,765,173-byte checkpoint** and **551,362,879-byte ZIP** were recovered
before shutdown with matching SHA-256. ZIP CRC, both camera models, raw final PLY,
all sixteen masks and CPU `weights_only` loading at step 6,999 passed; every tensor
was finite. Optimizer state is absent. Transfers stalled intermittently; verified
chunks were reused. An independent CLI ZIP attempt was cancelled after the
chunk watcher completed, rather than relied on after GPU deletion.

The L4 stopped after **23 min 13.3 s**, below the approved **40 min / 1.03-unit**
cap. Estimated task consumption was **0.596009 units** at the observed
1.54 units/hour; individual billing was not measured. Usage subsequently showed
zero assignments and zero consumption rate.

Native Chromium confirms walking, the featured ray push, a thrown ball and exact
replay at sampled states. The chair moves **0.817 m** and reaches **90.03°** tilt
over all recorded frames. The 8 s GIF is **1,131,733 bytes**, 480×270; the
1920×1080 MP4 is **3,909,411 bytes**. [Encoding receipt](demos/cafe-provenance.json).
This validates this reviewed public showcase, not automatic phone-video quality.


## Calibrated window-office trial, 9 October 2026

**Completed and recovered; not adopted for the README or live gallery.** Input:
144 undistorted Eyeful Tower `office_view2` photographs from cameras 19, 16 and
22, provided camera calibration, 100,000 sparse initialization points and sixteen
primary-camera segmentation photographs. This did not run VGGT or validate a
smartphone video. Source revision: `822afcc4555abf181cc8519b0753c165a8f2a427`.
GPU: **NVIDIA L4, 23,034 MiB**, driver 580.82.07. SAM 3 access and native bf16
preflight passed. The input ZIP's remote SHA-256 matched its local receipt.

| Stage | Measured seconds |
|---|---:|
| Fresh GPU environment setup | 579.247 |
| gsplat, 7,000 steps | 262.494 |
| SAM 3, sixteen photographs, `chair,cushion`, seed 11 | 39.823 |
| Automatic world assembly | 8.267 |
| Core stages combined | 310.584 |
| Enclosing preparation/run/backup preparation, excluding setup | 340.766 |
| Local manual two-view assembly diagnostic | 8.571 |

Training ended with **1,108,601 Gaussians**. The automatic world kept 1,105,822,
with **3 detected fragments, 4,617 colliders and 3 support patches**. These object
counts do not mean three complete movable chairs. Held-out metrics were PSNR
**20.9621 dB**, SSIM **0.800568** and LPIPS **0.380820**, for this trial only.
Photograph inputs and the schedule differ from the published VGGT window office.

The final checkpoint (261,632,757 bytes, step 6,999) and world ZIP (521,971,024
bytes) passed SHA-256/CRC checks before shutdown. Both recovered camera models
matched their prepared inputs byte for byte. CPU `weights_only` loading verified
all six finite Gaussian tensors; the checkpoint has no optimizer state and is
not an exact training-resume snapshot. Raw final PLY and sixteen segmentation
masks are retained locally.

The dedicated L4 stopped after **1,059.022 s (17 min 39 s)** measured from the
pre-allocation timer. At the observed 1.54 units/hour this is **an estimated
0.4530 units**, below the approved 25-minute/0.65-unit cap; individual billing
was not measured. Colab subsequently reported **0 assignments and 0.00 units/hour**.
Setup, transfer, local review and browser recording are separate from core stage
timing. No budget extension or further allocation was used.

Visual review rejected the trial: TV/window floaters remained, broad pruning
created dark gaps, and the manually associated chair failed visible movement and
tipping. Current README media and live assets were preserved.
[Detailed receipt](runs/quality/lounge-calibrated-20261009.json)
· [Actual comparison images and limitations](quality.md#window-office-calibrated-trial-not-adopted-9-october-2026).


## CPU window-office chair repair, 9 October 2026

This reuses the saved calibrated trial's PLY, camera models and SAM masks.
No GPU runtime or paid resources were started, and train/SAM timings remain those
of the earlier run rather than new measurements. Three manually associated mask
views were used for assembly; the 144-view training data was unchanged.

The experimental connected-voxel assembly took **11.100 s** on CPU. The subsequent
production implementation produced byte-identical background/object PLYs and
identical physical geometry. It retained **1,106,172 Gaussians**, with **1 reviewed
movable chair, 4,695 colliders and 1 support patch**. This is not automatic
multi-object segmentation validation. Distances depend on the assumed 1.5 m
camera-height scale.

Recording used the local **GTX 1660 Ti**, native Chromium/D3D11, fixed physics
step 1/60 s. All **120 frames** (8 s, 15 fps, 960×540) were captured in **48.910 s**,
including browser loading. The chair moved **1.202 m** and reached **140.92°**
maximum tilt over recorded frames. Two replays matched at sampled states.
The GIF is **1,187,731 bytes**, 480×270; the diagnostic MP4 is **1,341,045 bytes**,
960×540. These are diagnostic media, not a replacement for the existing hero.

Background blur remains unacceptable for adopting the reconstructed room. The
repair is available as an optional object filter and an explicit mask-review tool.
[Evidence and limitations](quality.md#window-office-chair-extraction-repair-9-october-2026)
· [Measured receipt](runs/quality/lounge-chair-repair-20261009.json).

## Local window-office checkpoint refinement, 9 October 2026

All runs used the recovered calibrated 144-photograph scene, the original
step-6,999 checkpoint and sorted split (126 train / 18 validation).
Hardware: **GTX 1660 Ti, 6GB**, WSL Ubuntu 22.04, PyTorch **2.9.1+cu128**, pinned
gsplat **1.5.3**, CUDA toolkit **12.8**. This is refinement with fresh Adam and
fixed Gaussian count/cameras, not a new end-to-end video run.

| Trial | Added steps | Best step | Best mean PSNR (dB) | Run time (s) |
| --- | ---: | ---: | ---: | ---: |
| Original checkpoint, local evaluation | 0 | 0 | 20.9642 | — |
| RGB-only, position LR 0.000016 | 1,000 | 250 | 21.0007 | 157.125 |
| Sparse-depth weight 0.05, LR 0.000016 | 1,000 | 750 | 21.1272 | 177.358 |
| Sparse-depth weight 0.5, LR 0.000016 | 2,000 | 2,000 | 21.3906 | 341.102 |
| Sparse-depth weight 0.5, LR 0.00016 | 2,000 | 2,000 | 21.4842 | 361.575 |

Run time includes source evaluation, training, periodic evaluation and checkpoint/
PLY exports; it excludes input loading and setup. Images retained their prepared
dimensions, with `--patch-size 2048 --max-size 1118`. The selected run's peak
**allocated Torch CUDA memory** was **1,531.119 MiB**. Native gsplat compilation
took **617 s** separately. Neither establishes full VGGT/SAM feasibility here.

The selected diagnostic has **1,106,182 Gaussians, 1 reviewed chair, 4,481 static
colliders and 1 support patch**. Browser recording captured 120 frames (8 s,
15 fps, 960×540) in **60.543 s**, including loading, using native Chromium/D3D11
on the same GPU. No paid resource was allocated.

Results were validation-selected; they are not independent test metrics or speed
guarantees. Visual review rejected README/gallery adoption.
An additional camera-neighborhood pruning probe removed 17,936 Gaussians and
scored 22.0779 dB / 0.813094 SSIM, but left severe bright occlusion in one view.
It was not adopted or browser/physics-validated; the recording above uses the
unpruned refined checkpoint. Its radius is a free-space heuristic, not a measurement.
[Evidence and unresolved blur](quality.md#local-window-office-refinement-9-october-2026)
· [Full receipt](runs/quality/lounge-refinement-20261009.json).

## Local interior cleanup and refinement, 9 October 2026

This continues the saved window-office checkpoint, not a new video reconstruction.
The same 144 calibrated public photographs, 126/18 split, reference points and
local **GTX 1660 Ti (6GB)** were used. No Colab allocation or SAM inference ran.

| Operation | Measured seconds |
| --- | ---: |
| CPU support filter, PLY/mask export and hashes | 4.228 |
| 500-step warm start, evaluations and checkpoint/PLY exports | 88.368 |
| Final 120-frame browser capture, including loading | 38.070 |

Refinement timing excludes checkpoint/image loading and setup. World assembly,
SPZ packaging and encoding are separate from these values; no combined
end-to-end runtime is claimed. Torch peak allocated CUDA memory for refinement
was **1,485.897 MiB**, excluding driver/non-Torch allocations. Original poses and
SAM masks were reused, so their old stage times are not new measurements.

Support filtering removed **34,071** Gaussians, leaving **1,074,530** for
refinement. Final assembly kept **1,072,039**, with **1 reviewed movable chair,
4,394 static colliders and 1 support patch**. Mean validation PSNR/SSIM changed
from **21.4842 / 0.805519** to **22.9816 / 0.824828** after filtering, then
**23.1516 / 0.830077** after 500 additional steps. Results were selected on the
same validation split, not an independent test set.

The final 8 s recording uses 15 fps, 960×540, native Chromium/D3D11 and fixed
1/60 s physics. The chair moved **0.672 m**, maximum recorded tilt **174.05°**;
two replays matched at eight sampled states. GIF: **1,105,676 bytes**, 480×270;
MP4: **1,290,700 bytes**, 960×540. Window haze remains, and this diagnostic was
not adopted for the current README/gallery.
[Visual comparison and limits](quality.md#window-office-interior-floater-cleanup-9-october-2026)
· [Full receipt](runs/quality/lounge-interior-cleanup-20261009.json).

## Local window cleanup and photo fallback, 9 October 2026

This continues the saved calibrated window-office trial on the local
**GTX 1660 Ti (6GB)**, with the same 144 photos and 126/18 training/validation
split. Existing poses, sparse points and manually associated SAM masks were
reused. No Colab allocation, paid resource or new SAM inference occurred.

| Operation | Measured seconds |
| --- | ---: |
| CPU observed-floor support filter, PLY/mask export and hashes | 4.516 |
| 500-step warm start, evaluations and checkpoint/PLY exports | 94.305 |
| CPU reviewed-photo baking, viewer refresh, export and hashes | 2.870 |
| Floor-cleaned browser capture, 120 frames including loading | 39.704 |
| Photo-window browser capture, 120 frames including loading | 38.347 |

Refinement excludes checkpoint/image loading and setup. Peak **allocated Torch
CUDA memory** was **1,418.756 MiB**, excluding driver/non-Torch allocations.
Assembly, SPZ packaging and encoding are separate; no combined video-to-world
time is claimed. The current baking CLI reproduced the recorded trial's texture,
background and world JSON hashes. The GIF uses **1,017,721 bytes**, 480×270, 8 s,
15 fps; MP4 uses **1,185,017 bytes**, 960×540.

Floor filtering removed **51,434** Gaussians and left **1,023,096** for refinement.
Raw Gaussian mean validation PSNR/SSIM changed from **23.1516 / 0.830077** to
**23.4102 / 0.828603** after filtering, then **23.9322 / 0.833482** after training.
These scores exclude the photo plane and use the same validation split that
guided selection; they are not independent final-test results.

The assembled diagnostic before photo replacement has **1,020,607 Gaussians**.
The photo-window version has **940,894**, with **1 reviewed movable chair,
4,047 static colliders, 1 support patch and 1 photo plane**. A fixed-step ray push
moved the chair **0.635 m**, maximum recorded tilt **130.87°**, initial settling
**0.006 m**. Two sampled replays matched exactly; all 120 recorded physics states
also matched between the floor-only and photo-window versions.

Photo texture represents the upper window and its exterior; true outdoor
parallax is absent. Lower-window haze and room streaks remain. This diagnostic
was not adopted for the README hero/live gallery. Full native tests passed
**124 tests**, with two skips; refinement, cleanup and photo-projection tests
passed **19 tests** in the Torch environment.
[Visual comparison and disclosure](quality.md#window-office-floor-footprint-and-photo-window-9-october-2026)
· [Full receipt](runs/quality/lounge-window-cleanup-20261009.json).
