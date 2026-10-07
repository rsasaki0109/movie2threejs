# Measured capture run

Eyeful Tower public apartment photograph sequence (not smartphone video). The full pipeline succeeded on an existing Colab L4. The browser preview showed substantial blur and floaters, so this reconstruction is not the hero asset. Physical interaction still needs validation. This is the no-BA run, with VGGT weights already cached. Environment setup and earlier failed attempts are excluded. SAM 3 weight loading is included in its stage.

Raw measured report: [run-20261007T081727Z.json](runs/run-20261007T081727Z.json).

- Status: `ok`
- Started (UTC): `2026-10-07T08:17:28.532111+00:00`
- GPU: NVIDIA L4, 23034 MiB, 580.82.07
- Video: `apartment-camera19.mp4`
- Video SHA-256: `adc56a3cf265dd420cc195254a83c69603a2f3fcd37288e52f68dc41da7e0989`

| Stage | Status | Measured seconds |
|---|---|---:|
| frames | ok | 1.235 |
| poses | ok | 34.839 |
| train | ok | 159.855 |
| segment | ok | 72.444 |
| world | ok | 5.037 |

Objects: 4. Static colliders: 7413.
Total for this invocation: 273.476 s (setup and reused stages excluded).

## Settings

```json
{
  "cmd": "all",
  "video": "/content/public_source/apartment-camera19.mp4",
  "scene": "/content/scene",
  "vggt_dir": "/content/playworld/.gpu/repos/vggt",
  "gsplat_dir": "/content/playworld/.gpu/repos/gsplat",
  "num": 24,
  "max_size": 1280,
  "ba": false,
  "pose_confidence": 1.5,
  "steps": 7000,
  "prompts": "chair,box,mug,bottle",
  "vggt_python": "/content/playworld/.gpu/envs/vggt/bin/python",
  "gsplat_python": "/content/playworld/.gpu/envs/gsplat/bin/python",
  "sam3_python": "/content/playworld/.gpu/envs/sam3/bin/python",
  "report": "/content/scene/run-20261007T081727Z.json",
  "start_at": "frames",
  "out": "/content/world",
  "masks": null,
  "eye_height": 1.5,
  "voxel": 0.1
}
```
