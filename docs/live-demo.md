# Live browser demo

The static gallery contains five actual Eyeful Tower captures reconstructed on an existing Colab L4: apartment workbench, meeting room, kitchen/cafe, window office and furnished room. No inference runs on the website. Each room includes its recorded timeline, preview, exact source and checksum manifest.

`demo-assets/workbench/` contains 28,007,776 bytes of `.splat` data, its `world.json` and a checksum manifest. This format retains Gaussian positions and scales as float32, quantizes color/opacity/rotation and omits view-dependent higher-order spherical harmonics. It reduces the original approximately 207 MB download; the hero media retains the original rendering. Dataset attribution and its MIT notice accompany the published site. Model restrictions still apply.

## Local preview

```bash
pip install -e '.[record]'
python scripts/build_gallery.py
python -m http.server -d _site 8000
```

Open http://localhost:8000 and choose a room from the gallery. The workbench enters at the hero's viewpoint. Desktop: WASD, mouse look, E push, click throw, Space jump, R reset, Esc menu. When the mouse is released, the **Knock the box over** button applies the same ray impulse as the recorded hero. It moves a real rigid body. Touch devices offer swipe look, directional buttons, push and throw. WebGL2 and a reasonably capable GPU are required; physical mobile hardware has not been benchmarked.

Run `python scripts/verify_site.py` after installing Playwright/Chromium to check the built site beneath `/movie2threejs/`. It verifies identical fixed-step replays, a box falling off the workbench, desktop mouse capture and keyboard walking, and touch entry/throw in mobile browser emulation. Screenshots and the result are written to `.cache/site-verification/`.

Run `python scripts/verify_gallery.py` to verify all five rooms, their asset checksums, two identical physics replays per room, the featured push, a thrown ball and both scripted and interactive walking. [Gallery validation](gallery-validation.json) records the checked results. Mobile controls were checked in browser emulation for the workbench; physical mobile hardware remains unmeasured.

The additional 8 s previews are actual 1280×720 fixed-step browser captures, encoded as MP4 and 480×270 looping GIFs below 2 MB each. Their recordings are [camera/event JSON files](../shots/gallery/), with encoding receipts under `docs/demos/`.

To regenerate the packed world from a local measured result, choose a fresh output directory:

```bash
python scripts/build_demo.py --world scenes/hero-result/world --profile shots/gallery/workbench-profile.json --out .cache/site-preview
```

## GitHub Pages

`.github/workflows/pages.yml` builds `_site` from committed assets and uploads that directory only. It includes viewer files, public world data, credits, the hero preview and recording timeline. Local caches, GPU dependencies and training output are excluded. No paid GPU is started.

The intended URL is `https://rsasaki0109.github.io/movie2threejs/`. It is **not live**. The user chose to keep the repository private and hold publication. GitHub returned `Your current plan does not support GitHub Pages for this repository`; the workflow skips private repositories. The gallery can be run locally without changing repository visibility.

Once this repository is public and Pages is configured with `build_type: workflow`, dispatch **Deploy live demo** or push a site change. After deployment, run:

```bash
python scripts/verify_site.py --url https://rsasaki0109.github.io/movie2threejs/demos/workbench/ --out .cache/site-live-verification
```

The deployed URL must pass this check before it is advertised as a working live demo in README or About.
