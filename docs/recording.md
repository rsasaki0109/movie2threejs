# Recording the README hero

The final GIF must come from a successfully reconstructed real capture. `shots/demo.json`
is a synthetic-room example only. Do not put synthetic checks in `docs/hero.gif`.

Install the capture tools and browser:

```bash
pip install -e '.[record]'
python -m playwright install chromium
```

After importing the real world returned by Colab, make and tune the shot:

```bash
python scripts/make_shot.py world shots/your-shot.json
python scripts/record.py --world world --shot shots/your-shot.json --out recordings/your-capture
python scripts/encode_hero.py --video scenes/public_source/apartment-workshop.mp4 --video-start 2 --frames recordings/your-capture --out docs --labels
```

Use `--software` for SwiftShader if the headless browser cannot use hardware rendering.
The initial smoke run used SwiftShader. The recorder now enables the browser GPU and
uses D3D11 on Windows; a complete 13 s synthetic capture used the local GTX 1660 Ti
and took 99.909 s (960×540, 195 frames). This is capture time, not reconstruction time.
Hardware enablement follows [Chromium's headless GPU guidance](https://chromium.googlesource.com/chromium/src/+/main/docs/gpu/using-gpu-hardware-in-headless-chrome.md).
The default capture is 1920×1080 at 30 fps; software rendering may take several minutes.

The provisional camera plan needs review: camera positions must stay inside the room,
the push ray must hit the selected movable object, and the ball must hit the tabletop object. It is
a cinematic camera path; interactive walking is separately controlled by Rapier's
character controller. Never infer collision correctness from a camera fly-through.

Shot JSON:

```json
{
  "version": 1, "fps": 30, "duration": 13,
  "camera": [
    {"time": 0, "position": [0, 1.5, 2], "target": [0, 0.5, 0]},
    {"time": 13, "position": [0, 1.5, 1.5], "target": [0, 0.5, 0]}
  ],
  "events": [{"time": 6, "type": "push"}, {"time": 9, "type": "throw"}]
}
```

Camera interpolation is smoothstep (or `"ease": "linear"` on the starting key).
Physics uses 1/60 s steps. Supported output rates are 15, 20, 30 and 60 fps.
`push` and `throw` use the same ray/ball behavior as the interactive viewer.
`walk` events set character keys, e.g. `{"time": 2, "type": "walk", "keys": ["KeyW"]}`;
an empty keys array stops walking. Events run on the first physics tick at or after their time.

`?record=shots/hero.json` loads a shot relative to the viewer URL; copy the shot into that
served directory when using the query directly. The Playwright script serves the selected
shot itself, captures only the canvas, and writes `capture.json` with every physics snapshot.
The same shot reloaded in the same build starts a new physics world and repeats deterministically.

The encoder creates a two-second source/world comparison, the remaining world footage,
and a half-second dissolve back to the first comparison frame. It writes a high-quality
H.264 MP4, a looping 960px GIF, and SHA-256 provenance. If [gifski](https://github.com/ImageOptim/gifski)
is on PATH, it uses gifski; otherwise it uses ffmpeg palettegen/paletteuse. You can also
pass `--gifski /path/to/gifski`. The encoder adjusts quality or palette/fps to fit
8,000,000 bytes and fails if it cannot meet that limit. Inspect the GIF and MP4 before
adding the README image. Preserve the dataset attribution and MIT notice alongside them.

`--video-start` selects the source excerpt. `--video-speed 0.5` plays one second
of that source over the two-second comparison. The playback speed is recorded in
the encoding receipt; it affects the source comparison, not recorded physics.

The synthetic encoding check produced a 960×540, 13.00 s looping GIF at 15 fps:
4,384,897 bytes with gifski quality 90. Its 1920×1080 MP4 was 6,989,834 bytes.
These validate capture and packaging only; they are not a real-data reconstruction or
the README hero. See [the validation record](synthetic-encoding-validation.json).

## Actual hero shot

`shots/hero.json` belongs to the measured Eyeful Tower workbench world; its coordinates are not reusable for another room. It lasts 13 s at 30 fps, with 50-degree vertical field of view. The intro is a source/world split-screen; then a camera approach, a box pushed from the workbench, a thrown ball, and a return to the opening view. The push specifies an action ray independently of the camera: `origin`, `target`, and impulse strength in m/s times body mass. The throw uses the viewer camera. Both use the interactive physics functions; there is no animation of the object's recorded pose.

Optional shot `fov` is 20–120 degrees; push `strength` is positive (interactive default 3). An event's `origin` and `target` must be provided together as distinct finite xyz vectors. `record.nextFrame(n)` can advance to a later frame for quick QA; every intervening physics tick and event is still simulated. Full exports call `nextFrame()` sequentially and write every frame.

Use `scripts/verify_hero.py` after encoding to check the actual asset sizes/durations, source hash, physical push, walking, ball creation, and matching physics at repeated sampled frames. [Hero validation](hero-validation.json) stores the results. Keep the [MIT notice](EYEFULTOWER-LICENSE.txt) with the media. The bottom cap and support patch are deliberately simple color approximations. Recording mode also hides the hosted viewer's navigation, footer and action controls, so they cannot cover canvas screenshots.

Reproduce the measured asset after building the gallery as described in
[local preview](live-demo.md#local-preview). The packaged data directory alone has
no viewer HTML; record the built viewer:

```bash
python scripts/record.py --world _site/demos/workbench --shot shots/hero.json --out recordings/hero
python scripts/encode_hero.py --video scenes/public_source/apartment-workshop.mp4 --video-start 0.666667 --video-speed 0.5 --frames recordings/hero --out docs --gifski /path/to/gifski --labels --font /path/to/font.ttf --gif-settings 10 58 58 58
```

The measured Windows capture used the installed Arial font and gifski 1.34.0. Encoding appends a short hold of the opening comparison so the final dissolve closes the loop; measured output durations are saved separately from the 13 s timeline.

The earlier 8 October capture, before box refinement, read a CRLF timeline; the repository stores that same JSON
with LF line endings. The original recording hash is retained as `shot_sha256`,
and `committed_shot_sha256` identifies the committed bytes. A replay of the LF file
matched all 390 physics states and both endpoint PNGs
([normalization check](runs/quality/workbench-timeline-normalization-validation.json)).
This was a replay check, not a second complete PNG capture.

The current box-refined capture reads the committed LF timeline directly, so its
`shot_sha256` identifies those exact bytes. The earlier CRLF normalization receipt
remains historical evidence; it is not needed to interpret the current capture.
