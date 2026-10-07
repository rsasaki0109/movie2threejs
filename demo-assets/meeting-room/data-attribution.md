# Calibrated meeting room attribution

Eyeful Tower **office1b**, cameras **19,16,22**, 192 undistorted photographs. This is a public photographic capture rig, not a phone recording or the author's room.

[Official dataset](https://github.com/facebookresearch/EyefulTower) / [MIT notice](EYEFULTOWER-LICENSE.txt) / [Exact photographs, calibration files and hashes](source.json).

Dataset revision: `06a01a4915afc872b893c20a025a0e14598c8478`. Credit: Linning Xu and coauthors, *VR-NeRF: High-Fidelity Virtualized Walkable Spaces*, SIGGRAPH Asia 2023.

This showcase uses dataset-provided COLMAP camera calibration **and sparse point initialization**. VGGT was not used. gsplat trained for 30,000 steps on an existing Colab L4. SAM 3 masks use sixteen camera-19 views. Browser review retained seed-frame-0 instance 1; other chairs remain static. The seed-frame-5 masks were rejected as partial or erroneous. [Measured run and selection](quality-report.json).

SPZ v3 retains quantized view-dependent color through SH3. Scale still assumes a 1.5 m camera height. Rigid bodies, hidden surfaces and collision geometry are approximate. This is a playable-world showcase, not evidence of automatic phone-video reconstruction. SAM 3 model terms apply separately from the MIT dataset license.
