# Colab CLI

The official [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)
supports Linux/macOS. On this PC it is installed as version 0.7.4 in WSL Ubuntu-22.04,
at `/root/.local/bin/colab`. Google authentication succeeded. An existing L4 assignment
was attached as `playworld-l4`, and project upload and remote preflight succeeded.
Preflight measured NVIDIA L4, 22,563 MiB PyTorch-reported GPU memory and native bf16. SAM 3
weight access was confirmed from the runtime (`approved`). A complete apartment reconstruction
ran through VGGT, gsplat, SAM 3 and world export. The short workbench BA reconstruction
took 419.079 seconds with cached weights, excluding setup. This is historical evidence;
the current hero uses the later calibrated 156-photograph workbench trial and a
reviewed 5,000-step checkpoint. See [benchmarks](benchmarks.md).
Setup, trainer help flags and SAM 3 API imports passed on 2026-10-07. No runtime was allocated.

Authenticate in your own PowerShell terminal:

```powershell
wsl -d Ubuntu-22.04 --exec /root/.local/bin/colab --auth oauth2 sessions
```

Open the URL printed by the CLI, approve in Google, and paste the authorization code
back into that terminal. Do not put the code or an HF token in chat or source files.
Then the agent can inspect existing sessions and account availability. Creating a new
runtime can consume compute units; these examples do not allocate one.

Browser-created assignments can appear with `[?]` in `colab sessions`. The CLI does not
resolve those endpoint IDs as session names. `scripts/colab_attach.py` uses the installed
CLI's assignment API and `StateStore.add()` to register exactly one existing endpoint;
it never calls the allocation API. Run it with the CLI environment's Python, specifying
the endpoint returned by `colab sessions` and an unused local name:

```bash
/root/.local/share/uv/tools/google-colab-cli/bin/python scripts/colab_attach.py --endpoint EXISTING_ENDPOINT --name playworld-l4
```

In WSL, from the project directory, use an existing suitable session (`SESSION`):

```bash
/root/.local/bin/colab exec -s SESSION --env PLAYWORLD_CLI_ACTION=check -f scripts/colab_job.py
/root/.local/bin/colab upload -s SESSION .cache/playworld-colab.zip /content/playworld-colab.zip
/root/.local/bin/colab exec -s SESSION --timeout 7200 --env PLAYWORLD_CLI_ACTION=setup -f scripts/colab_job.py
/root/.local/bin/colab exec -s SESSION --timeout 7200 --env PLAYWORLD_CLI_ACTION=run -f scripts/colab_job.py
/root/.local/bin/colab download -s SESSION /content/playworld-result.zip .cache/playworld-result.zip
```

The first command checks actual GPU hardware and approved SAM 3 access, without setup
or inference. It reads `HF_TOKEN` from the existing remote environment or Colab Secrets;
or the one-use `.playworld-hf-token` transfer file, which is removed after loading. If
the secret is unavailable in this runtime, the preflight reports that and the run
does not proceed. SAM 3 uses the same L4/A100/native-bf16 requirement as `run.sh`.

For the one-use credential transfer, authenticate locally and then run:

```bash
/root/.local/bin/uvx --from huggingface-hub hf auth login --format agent
python3 scripts/colab_hf_token.py --session SESSION
```

The HF CLI prints a device code to approve in the browser; no token needs to be pasted
into chat. The transfer uses a temporary file with mode 0600, removes the local transfer
file, and keeps the token out of CLI arguments/history. The remote preflight loads and
removes the transfer file. SAM 3 model approval and CLI login are separate requirements.

If HF login was performed on Windows rather than inside WSL, use that cached token
file explicitly. This transfer was used in the 8 October notebook check:

```powershell
wsl -d Ubuntu-22.04 --exec python3 /mnt/c/Users/rsasa/Workspace/movie2threejs/playworld/scripts/colab_hf_token.py --session SESSION --token-file /mnt/c/Users/rsasa/.cache/huggingface/token
```

Replace the project and user paths for your PC. The argument is a file path,
not the token itself. [Interactive notebook verification](colab-ui.md) records
which UI steps passed and which remain unverified.

The explicit `run` action unpacks the project, runs the shared setup and pipeline scripts,
and packages the world and measured evidence. On setup/pipeline failure it packages logs
without exporting a stale world. The wrapper never provisions or stops a runtime. Preserve
the result before rerunning, and release a task-owned runtime when the work finishes.
The `setup` action can install the isolated environments while weight approval is pending.
Do not interpret installation success as inference success.
After a successful setup, `--env PLAYWORLD_CLI_SKIP_SETUP=1` on the `run` command reuses
those environments. The wrapper requires the successful setup log, help checks and all
four environment snapshots before accepting that explicit request.

The existing browser-upload notebook remains available; its `files.upload()` cells should
not be executed through the CLI. The CLI entry point avoids those interactive cells.

## Chunked artifact downloads

Large exports were not fully recovered in the latest GPU reconstruction.
These helpers were prepared afterwards. Local binary fixtures and an
approximately 847 MB loopback HTTP transfer verify assembly, retry, resume and
corruption checks. A subsequent Colab CPU test recovered both synthetic artifacts
with matching whole-file hashes and ZIP integrity, using retries and higher
concurrency. Automatic recovery before deletion and a full GPU scene backup
remain **unverified**. They never allocate or stop a runtime.

Start final-checkpoint recovery as soon as training finishes, before previewing
the room. While the existing task-owned runtime is still connected, create
8 MiB chunks and SHA-256 manifests on that runtime:

```bash
python scripts/colab_pack.py /content/RUN/scene/gs/ckpts/ckpt_6999_rank0.pt /content/RUN/checkpoint-parts --step 6999
python scripts/colab_pack.py /content/RUN/playworld-result.zip /content/RUN/result-parts
```

Replace `/content/RUN` with the attempt directory printed by the notebook, and
use the actual final step/file for a different training length. Packaging
requires a fresh output directory. The ZIP must already have been created.
The helpers are local additions; upload them to an older pinned source if needed.

On the local PC, use native Python to download eight chunks concurrently by
default (up to 32 with `--workers 32`):

```bash
python scripts/colab_download.py --session SESSION --remote-directory /content/RUN/checkpoint-parts --out checkpoint-backup --expected-step 6999
python scripts/colab_download.py --session SESSION --remote-directory /content/RUN/result-parts --out world-backup
```

On Windows, the helper reads an existing official CLI session through WSL,
then transfers bytes with native HTTP. Adjust `--distro` and `--sdk-python` if
the CLI is installed elsewhere. Proxy credentials stay in process memory and
are excluded from transfer logs. Verified chunks are reused on a retry; corrupt
chunks are fetched again. The output becomes complete only after its size and
SHA-256 match the manifest. An existing different output is not overwritten.

Runtime deletion can interrupt these transfers. Before deleting the runtime,
verify ZIP integrity, load the final checkpoint on CPU and review the exported
world. Matching file hashes alone does not establish model loading or visual quality.

## Automatic backup watcher

The updated notebook calls `run.prepare_backups()` immediately after a successful
pipeline, before its preview cell. It publishes an atomic `backup-index.json`
as soon as the final checkpoint's chunks are ready, then packages and publishes
the world ZIP. A failed ZIP preparation retains the ready checkpoint. A retried
pipeline invalidates its prior index, so it cannot offer an earlier result.
This order was subsequently exercised with a new L4 reconstruction; its final
checkpoint and ZIP were fully recovered and checked before termination.
[GPU backup receipt](colab-gpu-backup-20261008.json).

Start this local command after the notebook prints the attempt directory, while
reconstruction is still running:

```bash
python scripts/colab_download.py --session SESSION --remote-index /content/RUN/backup-index.json --expected-step 6999 --out backup-ATTEMPT --wait-seconds 1800
```

Replace `SESSION`, `/content/RUN` and the expected final step. Use a fresh output
directory for each attempt. The command polls for the index, downloads each ready
artifact and records size/SHA-256 in `download-receipt.json`. Exhausted connection
retries return to polling and reuse verified chunks. The retry branch is locally
tested; the subsequent real GPU watcher completed without requiring that branch. If interrupted,
rerunning with the same output directory reuses verified chunks. It does not wait
for the preview or download iframe. The wait limit does not allocate, extend or
stop any runtime; GPU time/unit limits must still be enforced separately.

Successful notebook archives also retain available `scene/sparse/cameras.bin`,
`images.bin`, `points3D.bin`, `scene/pose-diagnostics.json` and
`scene/calibrated-source.json`. This allows later training-view comparisons in
the original camera frame. Image and weight directories are excluded; the final
checkpoint remains a separate backup. The older verified GPU archive did not
include these files. The new packaging passed local synthetic export/recovery
tests and has not yet been exercised in Colab.

## Native Windows execution fallback

In the 8 October backup attempt, WSL failed to reach the assigned runtime before
dispatching code. A native Windows `jupyter-kernel-client==0.9.0` connection then
executed the notebook's first six code cells: settings, source bootstrap,
preflight, setup, public input and reconstruction/backup preparation. This does
not exercise the two interactive preview/download cells in the Colab browser.
Those unchanged cells subsequently passed a separate CPU browser check with the
actual recovered GPU world; see [the UI record](colab-ui.md#actual-gpu-world-restored-in-colab).

The initial native console connection closed when its Windows encoding rejected
a tqdm character; the remote reconstruction continued and completed. The
[execution helper](../scripts/colab_execute.py) uses UTF-8 output, suppresses SDK
logs that can contain proxy credentials, and refuses an already busy kernel.
Busy-kernel refusal and a subsequent idle-kernel Unicode print passed on the
same L4. Inspect the remote job/log after any connection error before rerunning.

Use an already authenticated, task-owned session. For example, in PowerShell:

```powershell
py -3.12 -m venv .cache/colab-client
.cache/colab-client/Scripts/python.exe -m pip install jupyter-kernel-client==0.9.0
.cache/colab-client/Scripts/python.exe scripts/colab_execute.py --session SESSION --file notebooks/playworld_colab.ipynb --cells 6 --timeout 1800
```

Use the credential transfer above if Colab Secrets are unavailable. The helper
does not allocate, extend or stop a runtime. Its timeout closes the local
connection; it is not a GPU billing limit or a guarantee that remote work stops.
The existing-session lookup still uses the authenticated official CLI in WSL,
while kernel HTTP/WebSocket traffic uses native Windows networking.

## Transfer validation without inference

The measured local check used **441,186,741 checkpoint-shaped bytes** and a
**405,816,777-byte ZIP**, all synthetic. Eight native HTTP workers recovered both
in **29.078 s** over Windows loopback. ZIP integrity, final-file hashes, reuse
of a verified chunk, replacement of a corrupt chunk and retry of an intentionally
truncated HTTP response passed. Preparation took **78.906 s**.
[Receipt](colab-transfer-local-20261008.json).
This is not Colab throughput, model loading or a recoverable GPU scene.

To reproduce locally (requires about 4.3 GB free disk and a fresh work directory):

```bash
python scripts/verify_colab_transfer.py --workdir .cache/transfer-check --out .cache/transfer-check.json
```

For a subsequent Colab network check, use an approved task-owned **CPU** runtime
and its source scripts; no GPU, model installation or HF credentials are needed:

```bash
python scripts/verify_colab_transfer.py --prepare-only --workdir /content/transfer-check --out /content/transfer-preparation.json
```

Run the local backup watcher against the printed `backup_index`. Verify both
artifacts locally before stopping that runtime. This real-Colab network check
was subsequently performed, with the limits recorded below. These commands
do not authorize allocating paid resources.

## Colab CPU transfer result, 8 October 2026

An approved dedicated standard CPU prepared synthetic artifacts in **19.843 s**.
The **441,186,741-byte checkpoint-shaped file** and **405,816,730-byte ZIP**
were fully recovered through Colab's files proxy to native Windows Python.
Both SHA-256 values matched the remote manifests; ZIP CRC checks and the
synthetic payload marker passed. No GPU, model setup or HF credentials were used.
[Execution receipt](colab-transfer-cpu-20261008.json).

Eight workers were slow. The 32-worker index watcher retained verified chunks
but failed on `part037`; a separate eight-worker checkpoint retry and concurrent
32-worker ZIP download recovered the remaining data. The CLI now accepts up to
32 workers. Increasing concurrency can produce connection failures; it is not
a guarantee of faster or uninterrupted transfer.

The CPU was terminated after **19 min 19.6 s**, within the 20-minute limit.
At the inferred **0.08 units/hour**, estimated task consumption was **0.025770
units**, below the 0.03-unit cap; individual billing was not measured. Other GPU
assignments were left running. Already-started downloads continued after shutdown;
the checkpoint and ZIP's final local writes were at 07:56:34 and 07:57:30 UTC,
after the termination log completed at 07:54:16 UTC.

This proves complete recovery of synthetic bytes with manual retry, not actual
model loading, stable completion of the index watcher or backup before deletion.
Do not rely on transfer continuation after shutdown. For a GPU run, verify both
archives/checkpoint loading while the runtime is still available.
