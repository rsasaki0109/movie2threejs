# Measured capture run

Partial validation: Eyeful Tower public apartment photograph sequence, not a smartphone video. Only frame extraction ran on this PC; no GPU inference has run.

- Status: `partial`
- Started (UTC): `2026-10-07T04:16:42.899972+00:00`
- GPU: NVIDIA GeForce GTX 1660 Ti, 6144 MiB, 596.36
- Video: `apartment-camera19.mp4`
- Video SHA-256: `adc56a3cf265dd420cc195254a83c69603a2f3fcd37288e52f68dc41da7e0989`

| Stage | Status | Measured seconds |
|---|---|---:|
| frames | ok | 10.004 |
| poses | not run | — |
| train | not run | — |
| segment | not run | — |
| world | not run | — |

Objects: not measured. Static colliders: not measured.

## Settings

```json
{
  "num": 24,
  "max_size": 1280,
  "scope": "local frame extraction only; no GPU inference"
}
```
