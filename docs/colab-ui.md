# Interactive Colab verification

On 8 October 2026, Chrome imported `notebooks/playworld_colab.ipynb` into
Google Drive and ran its public video/VGGT recipe through world assembly.
The saved notebook output contains a [successful final report](colab-ui-run-20261008.json).

## What ran

- Project ZIP upload using the **Files sidebar**, settings and project extraction.
- GPU and SAM 3 access preflight: NVIDIA L4, native bf16, approved model access.
- Installation of the three isolated GPU environments: **601.612 s**.
- Public source download: **7.728 s**.
- Reconstruction: **745.670 s**, including initial model downloads; **8 movable
  objects, 2,962 static colliders and 1,982,706 exported Gaussians**.

This is an Eyeful Tower capture-rig photograph sequence encoded as a video,
not a smartphone recording. Stage times, settings and input hash are in
[benchmarks](benchmarks.md#interactive-notebook-run-8-october-2026).
The export has not received the manual visual review used for the README hero.

The approved dedicated L4 was stopped within its 30-minute limit, about
18 seconds after reconstruction completed. An intermediate checkpoint was
downloaded and loaded on CPU before shutdown: step **4,665**, **1,983,640
Gaussians**, **468,142,005 bytes**, SHA-256
`d4e24a7fa4f44e30dc110ad7d6a228956045d636b771a70cb2e04c2e050440a0`.
This is not the final checkpoint or a complete recoverable scene. The final
world and result ZIP were not downloaded before shutdown.

## What remains unverified

The notebook's browser preview and result ZIP download cells were not run.
The Colab Secrets grant flow was not verified: `HF_TOKEN` was absent, so the
run used the existing authenticated account's one-use token transfer described
in [CLI instructions](colab-cli.md). No token was printed or committed.
The optional phone-video upload path also remains unverified.

The rendered training output became large enough to make the browser
unresponsive. The updated runner now displays progress at most once every
30 seconds, plus completion, while retaining every progress update in its disk
log. This change passed local regression tests but has not been rerun on Colab.

## Upload and runtime checks

Top-level notebook and project uploads worked after enabling file URL access
for the ChatGPT extension on the correct PC. Selecting a project inside the
`files.upload()` output iframe timed out in browser automation; the notebook
now directs users to upload the project ZIP using the Files sidebar.

The existing-runtime URL from `colab url` did not attach the imported Drive
notebook to the intended L4 in this check. Connecting instead allocated a T4;
it was stopped immediately after detecting the mismatch, before installation
or inference. The existing L4 belonged to another project and was left intact.
A dedicated L4 was then allocated with explicit approval. Confirm the actual
GPU and active assignments before installation or model execution.

## Attempt isolation and local checks

Each attempt lives under its own `/content/playworld-runs/` directory,
printed after project loading. Loading the identical project ZIP again is
allowed; a different ZIP is rejected without replacing the existing project.
Weight access is checked before installation. Logs and `job.json` preserve
completion and failures. The final download cell constructs a fresh ZIP,
including a world only when the current pipeline completed. Preview requires
completion and uses a free server port; rerunning it shuts down only its own
previous server.

Local verification: **53 Python tests passed, 1 optional test skipped**.
Regression checks cover a failed retry after a successful world, failure before
input is available, independent attempts, an exit without a world manifest,
and throttled progress with a complete disk log. These tests do not substitute
for the remaining Colab browser checks.
