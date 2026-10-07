# Furnished room capture attribution

Eyeful Tower, **raf_furnishedroom**, camera 19. This is a public photographic capture-rig sequence, not a phone recording or the author's room.

[Dataset repository](https://github.com/facebookresearch/EyefulTower) / [Dataset MIT notice](EYEFULTOWER-LICENSE.txt) / [Exact source, trim and hashes](source.json).

The dataset's MIT notice was checked at revision `06a01a4915afc872b893c20a025a0e14598c8478`. Credit: Linning Xu and coauthors, *VR-NeRF: High-Fidelity Virtualized Walkable Spaces*, SIGGRAPH Asia 2023.

VGGT, gsplat and SAM 3 ran on an existing Colab L4. Gaussian splats were packed with DC color and quantized opacity/rotation. Higher-order spherical harmonics were omitted. Scale, rigid bodies and hidden surfaces are approximate. VGGT weight restrictions and SAM 3 model terms apply separately.
