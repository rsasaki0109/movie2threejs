# Measured public room assets

These `.splat` files are compact derivatives of five measured Eyeful Tower reconstructions: apartment workbench, meeting room, kitchen/cafe, window office and furnished room. Each folder includes its exact source and hash manifest. See [source and attribution](../docs/data-attribution.md), [dataset MIT license](../docs/EYEFULTOWER-LICENSE.txt), and [benchmark receipts](../docs/benchmarks.md).

Each `manifest.json` records the source world hash, Gaussian counts, sizes and packed asset hashes. Packing does not infer new camera poses or hulls. Browser review selected one stable meeting-room chair and six cafe instances; the window office was rebuilt with explicit capture bounds before packing. Each room adds its checked spawn, viewpoint and featured push action.

The compact 32-byte `.splat` format drops higher-order spherical harmonics and quantizes color, opacity and quaternion rotation. Model terms, including default VGGT non-commercial weights, apply separately from the dataset license.
