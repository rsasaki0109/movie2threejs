# Apartment workbench capture attribution

This room is reconstructed from 156 Eyeful Tower apartment photographs from cameras 19,16,22, using provided COLMAP calibration and sparse initialization. VGGT is bypassed. The ordinal interval is 60:112 per camera, with an exclusive stop. This is a public photographic capture-rig sequence, not a phone recording or the author's room. The hero's source comparison uses a camera-19 excerpt starting at 5.0 seconds with measured duration 4.250 seconds.

[Exact source and hashes](source.json) / [Dataset MIT notice](EYEFULTOWER-LICENSE.txt) / [Official dataset](https://github.com/facebookresearch/EyefulTower).

The MIT dataset notice was checked at revision 06a01a4915afc872b893c20a025a0e14598c8478. Credit: Linning Xu and coauthors, VR-NeRF: High-Fidelity Virtualized Walkable Spaces, SIGGRAPH Asia 2023.

The compact world retains 1,110,159 Gaussians, one reviewed movable box and 2,834 static colliders. Rendered comparison selected the 5,000-step checkpoint from a full 30,000-step training trial. Other detected objects stay static. SPZ v3 retains quantized view-dependent color through SH3. Scale, masks, collisions and hidden surfaces are approximate; blurry floor regions remain. Default VGGT weights have non-commercial terms, although this calibrated example does not use them. SAM 3 requires approved access and its own model terms.
