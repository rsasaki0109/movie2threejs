# Interactive Colab verification

## Automatic source loading

The current notebook fetches its source automatically from a pinned public
GitHub commit, with no project ZIP upload. Only `src/`, `scripts/`, `notebooks/`
and root files are checked out; the large gallery exports are not fetched.
The same unchanged revision can be loaded again without replacing environments
or results. A different revision, a ZIP-based existing project or edited source
is rejected instead of being overwritten. A failed fetch leaves no partial
project directory. Each attempt records the repository and revision in `job.json`.

A fresh fetch from public GitHub and repeat loading passed locally at commit
`9ed3cfc910379a876d4149944bb1f7c788f67a39`: **56 source files, 252,196 bytes**,
excluding Git metadata and gallery assets. Local tests: **56 passed, 1 optional
test skipped**. The subsequent Colab CPU check also loaded source automatically,
including the preview fix at `4d74f58073c8dc81c0889f738b85de0fcaca9590`.

The reconstruction measurements below came from the earlier ZIP-upload version.
They do not establish a complete first run of the updated notebook.

Local checks also served the existing public workbench through the notebook's
HTTP server and exported a **20,148,787-byte ZIP**. Archive integrity, the world
JSON and both SPZ checksums matched the input. These are local HTTP/ZIP checks,
not a Colab browser download or a new reconstruction.
[Local validation receipt](colab-bootstrap-local-validation.json).

## CPU browser preview and download check

On 8 October 2026, an approved dedicated standard CPU runtime ran the main
notebook's source-fetch, preview and download cells in Chrome. A separate
preparation cell loaded the **existing public workbench export** and checked
its world and SPZ hashes. No reconstruction, model installation, GPU or Hugging
Face credentials were used. [Validation receipt](colab-cpu-ui-run-20261008.json).

The room rendered through Colab's port proxy. Pointer lock failed with
`WrongDocumentError`, so the Colab preview now uses `?controls=drag`: drag to
look, WASD to walk, F to throw, E to push, R to reset and Esc to pause.
Chrome confirmed entry, a changed view after dragging and a visible thrown
ball. Local Chromium regression checks cover walking, drag look, throwing,
pause/resume/reset and the default pointer-lock mode of a downloaded world.

The final browser download was **20,149,354 bytes**. ZIP integrity, the world
and both SPZ hashes passed; its viewer bytes match the pinned source commit.
The archive and visual evidence were saved locally before terminating the
runtime. The dedicated CPU was stopped within the approved 15-minute limit;
the CLI then reported no active sessions. This account displayed **0.08 units
per hour** for the CPU; actual compute-unit consumption was not measured.

## Earlier reconstruction check

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

The updated notebook has not been rerun end to end through GPU reconstruction.
The CPU browser check above verifies those cells with a prebuilt world, not
preview/export of a newly reconstructed result from the earlier L4 run.
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
`files.upload()` output iframe timed out in browser automation; the earlier
ZIP-loading notebook used the Files sidebar. The current notebook replaces
that project upload with automatic source fetching.

The existing-runtime URL from `colab url` did not attach the imported Drive
notebook to the intended L4 in this check. Connecting instead allocated a T4;
it was stopped immediately after detecting the mismatch, before installation
or inference. The existing L4 belonged to another project and was left intact.
A dedicated L4 was then allocated with explicit approval. Confirm the actual
GPU and active assignments before installation or model execution.

## Attempt isolation and local checks

Each attempt lives under its own `/content/playworld-runs/` directory,
printed after project loading. Automatic source reuse checks the pinned
revision and rejects source changes without replacing the existing project.
Weight access is checked before installation. Logs and `job.json` preserve
completion and failures. The final download cell constructs a fresh ZIP,
including a world only when the current pipeline completed. Preview requires
completion and uses a free server port; rerunning it shuts down only its own
previous server.

Earlier local verification: **53 Python tests passed, 1 optional test skipped**.
Regression checks cover a failed retry after a successful world, failure before
input is available, independent attempts, an exit without a world manifest,
and throttled progress with a complete disk log. These tests do not substitute
for the remaining GPU first-run and Secrets checks.
