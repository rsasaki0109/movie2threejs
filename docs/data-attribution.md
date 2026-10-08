# Public capture attribution

Selected input: **Eyeful Tower, apartment, camera 19**.

This is the dataset's single-camera video visualization of sampled photographs taken by
a multi-camera capture rig. It is a real apartment, but not the user's room and not a
smartphone recording. Any GIF/MP4 created from it must identify it as public room capture.

Source: [official dataset repository](https://github.com/facebookresearch/EyefulTower).
The [README at the inspected revision](https://github.com/facebookresearch/EyefulTower/blob/06a01a4915afc872b893c20a025a0e14598c8478/README.md)
states that all content was relicensed under MIT on 11 March 2026.
The [MIT notice](EYEFULTOWER-LICENSE.txt) must accompany redistributed derived assets.
The downloader saves the source URL, revision, SHA-256 and attribution in `source.json`.

Credit: Linning Xu and coauthors, **VR-NeRF: High-Fidelity Virtualized Walkable Spaces**,
ACM SIGGRAPH Asia 2023, [DOI: 10.1145/3610548.3618139](https://doi.org/10.1145/3610548.3618139).

The capture file is 1368×2048, 12 fps, 176 frames, 14.667 seconds, 11,665,667 bytes.
SHA-256: `adc56a3cf265dd420cc195254a83c69603a2f3fcd37288e52f68dc41da7e0989`.
These properties were measured locally with ffprobe and SHA-256 on 7 October 2026.

Mip-NeRF 360 and Tanks and Temples were considered but not selected for redistributed
hero assets: an explicit redistribution license was not established for Mip-NeRF 360,
and Tanks and Temples' license page contains conflicting redistribution clauses.
The selected dataset license does not remove VGGT or SAM 3 weight restrictions.

## Hero derivative

The hero's source comparison uses the workbench excerpt `apartment-workshop.mp4`: start 5.0 s, requested duration 4.2 s, measured 4.250 s, 51 frames, 9,199,102 bytes. SHA-256: `a0a387a1af7f8483d8c5d13861dc17907054784f090a4c26e8370624b4e4263e`.

The current reconstructed world uses **156 undistorted photographs from cameras 19,16,22**, ordinal interval **60:112** independently per camera, dataset-provided COLMAP calibration and 100,000 sparse initialization points. **VGGT is bypassed.** The photograph manifest preserves exact URLs and hashes in [the packaged source](../demo-assets/workbench/source.json). Training ran for 30,000 steps; rendered comparison selected the checkpoint at 5,000 steps. This is a calibrated public showcase, not evidence of automatic smartphone conversion. The historical 24-image video/VGGT result is retained in the benchmark records.

The first two seconds of the hero show this public source beside the browser world. Text identifies it as **PUBLIC ROOM CAPTURE**. Camera movement, the pushed box, thrown ball and their collisions come from the browser viewer. The reviewed box has a simple colored horizontal bottom cap; other hidden object faces use colored convex interiors. These approximate unobserved geometry without inventing photographic texture.

The exact source trim is shared by the notebook and `scripts/download_public.py --workshop`. The source offset used for the split-screen comparison is recorded in `hero-provenance.json`. Keep `EYEFULTOWER-LICENSE.txt` with `hero.gif`, `hero.mp4` and any redistributed derivatives.

## Additional room gallery

All four additional examples use the same MIT-licensed Eyeful Tower dataset. Kitchen/cafe, window office and furnished room use camera-19 photograph-sequence videos with VGGT-estimated poses; no provided camera calibration was used for those three.

The current **meeting room** uses 192 undistorted photographs from cameras **19,16,22**, dataset-provided COLMAP camera calibration **and sparse point initialization**, with 30,000-step gsplat training and SAM 3 masks. It does not use VGGT and does not demonstrate automatic video-pose estimation. The previously committed video/VGGT run is preserved as historical benchmark evidence.

| Demo | Dataset | Source trim (start / requested seconds) | Exact provenance |
|---|---|---|---|
| Meeting room | office1b | 192 calibrated photographs; no video trim | [Photographs, calibration and hashes](../demo-assets/meeting-room/source.json) |
| Kitchen/cafe | kitchen | 14 / 6 | [Source and hashes](../demo-assets/cafe/source.json) |
| Window office | office_view2 | 0 / 5.5 | [Source and hashes](../demo-assets/lounge/source.json) |
| Furnished room | raf_furnishedroom | 0 / 3.5 | [Source and hashes](../demo-assets/furnished-room/source.json) |

Each packaged viewer includes its room-specific source, attribution and MIT notice. Keep the dataset notice with all previews and exports. Model-weight licenses apply separately. The three Explore/Push/Throw GIF excerpts are from the original apartment workbench, rather than additional spaces.
