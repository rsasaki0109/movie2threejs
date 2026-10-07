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
