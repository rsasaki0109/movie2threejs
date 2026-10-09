# Kitchen & cafe capture attribution

Eyeful Tower, **kitchen**, cameras **19,16,22**: 144 undistorted photographs from ordinal interval **168:240**, with provided camera calibration and 100,000 sparse initialization points. This is a public photographic capture-rig sequence, not a phone recording or the author's room.

[Dataset repository](https://github.com/facebookresearch/EyefulTower) / [Dataset MIT notice](EYEFULTOWER-LICENSE.txt) / [Exact photographs, calibration and hashes](source.json).

The dataset's MIT notice was checked at revision `06a01a4915afc872b893c20a025a0e14598c8478`. Credit: Linning Xu and coauthors, *VR-NeRF: High-Fidelity Virtualized Walkable Spaces*, SIGGRAPH Asia 2023.

gsplat and SAM 3 ran on a dedicated Colab L4; VGGT was bypassed. One foreground chair was selected manually. Gaussian splats are packed as SPZ v3, retaining quantized view-dependent color through SH3. The chair's colored upper interior, clipped at assumed height 0.38 m, and its stationary floor patch are approximate unseen surfaces. Collision geometry is unchanged by that visual correction. Scale remains assumed; floor and vegetation blur remain. SAM 3 model terms apply separately.
