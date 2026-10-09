# Live browser demo

The static gallery contains five actual Eyeful Tower captures reconstructed on an existing Colab L4: apartment workbench, meeting room, kitchen/cafe, window office and furnished room. No inference runs on the website. Each room includes its recorded timeline, preview, exact source and checksum manifest.

The workbench, meeting room and kitchen/cafe are calibrated showcases using 156, 192
and 144 photographs, provided camera poses and sparse initialization; none uses VGGT.
Window office and furnished room use estimated VGGT poses. Selected viewpoints avoid some of the worst artifacts;
walking to other views can still reveal blur and incomplete coverage.

`demo-assets/workbench/` contains 20,079,956 bytes of SPZ v3 data, its `world.json` and a checksum manifest. It keeps the 5,000-step checkpoint from a 30,000-step trial, one manually reviewed box, and 2,834 static colliders. This quantized format retains view-dependent spherical harmonics through SH3. The hero and packaged viewer use the same SPZ export. A local CPU refinement removes 707 box splats with conservative mask/size/support checks and adds an approximate horizontal bottom, while keeping the original collision body and support patch. Conservative background-volume cleanup retains thin surfaces; it does not reconstruct unseen geometry. Dataset attribution and the MIT notice accompany each room. Model restrictions still apply.

## Local preview

```bash
pip install -e '.[record]'
python scripts/build_gallery.py
python -m http.server -d _site 8000
```

Open http://localhost:8000 and choose a room from the gallery. The workbench enters near the hero's opening view, at a reviewed clear walking position. Desktop: WASD, mouse look, E push, click throw, Space jump, R reset, Esc menu. When the mouse is released, the **Knock the box over** button applies the same ray impulse as the recorded hero. It moves a real rigid body. Touch devices offer swipe look, directional buttons, push and throw. WebGL2 and a reasonably capable GPU are required; physical mobile hardware has not been benchmarked.

Run `python scripts/verify_site.py` after installing Playwright/Chromium to check the built site beneath `/movie2threejs/`. It verifies identical fixed-step replays, a box falling off the workbench, desktop mouse capture and keyboard walking, and touch entry/throw in mobile browser emulation. Screenshots and the result are written to `.cache/site-verification/`.

Run `python scripts/verify_gallery.py` to verify all five rooms, their asset checksums, two identical physics replays per room, the featured push, a thrown ball and both scripted and interactive walking. [Gallery validation](gallery-validation.json) records the checked results. Mobile controls were checked in browser emulation for the workbench; physical mobile hardware remains unmeasured.

The additional 8 s previews are actual fixed-step browser captures, encoded as MP4 and 480×270 looping GIFs below 2 MB each. The current kitchen/cafe was captured at 1920×1080; exact dimensions and hashes are recorded in each encoding receipt under `docs/demos/`. Their recordings are [camera/event JSON files](../shots/gallery/).

To regenerate an SPZ world, install the optional encoder with `bash scripts/setup_spz.sh` in Linux/Colab. Use your freshly assembled PLY world as input and choose fresh asset and site directories:

```bash
python scripts/build_demo.py --world reviewed_world --profile shots/gallery/workbench-profile.json --format spz --assets .cache/spz-assets --out .cache/site-preview
```

## GitHub Pages

`.github/workflows/pages.yml` builds `_site` from committed assets and uploads that directory only. It includes viewer files, public world data, credits, the hero preview and recording timeline. Local caches, GPU dependencies and training output are excluded. No paid GPU is started.

The gallery is live at **https://rsasaki0109.github.io/movie2threejs/**.
The repository was made public with approval on 8 October 2026. The
[initial Pages deployment](https://github.com/rsasaki0109/movie2threejs/actions/runs/37719672967)
built the committed five-room gallery without GPU inference.

Pages uses `build_type: workflow`. Dispatch **Deploy live demo** or push a site
change to deploy again. After deployment, run:

```bash
python scripts/verify_site.py --url https://rsasaki0109.github.io/movie2threejs/demos/workbench/ --out .cache/site-live-verification
python scripts/verify_gallery.py --url https://rsasaki0109.github.io/movie2threejs/ --out .cache/gallery-live-verification
```

The deployed gallery passed asset checksums, two identical physics replays per
room, featured pushes, thrown balls and keyboard walking in all five rooms.
[Deployed gallery results](pages-gallery-validation.json), refreshed after the
[9 October kitchen deployment](https://github.com/rsasaki0109/movie2threejs/actions/runs/37864859297).
The calibrated kitchen also passed the explicit 60-degree tipping threshold
(sampled peak 90 degrees). The workbench also
passed pointer lock and touch walking/look/throw in Chromium mobile emulation,
with no page errors or failed requests. [Workbench results](pages-workbench-validation.json).
Physical mobile hardware remains untested.

The README's three-command synthetic-room start was checked from a fresh virtual
environment installed directly from the public GitHub repository, including
world generation and browser rendering. [Installation receipt](public-install-validation.json).
