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
These helpers were prepared afterwards. Local binary fixtures verify assembly,
resume and corruption checks; a complete transfer from a real Colab runtime
with them remains **unverified**. They never allocate or stop a runtime.

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

On the local PC, use native Python to download up to eight chunks concurrently:

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

Runtime deletion interrupts these transfers. Before deleting the runtime,
verify ZIP integrity, load the final checkpoint on CPU and review the exported
world. Matching file hashes alone does not establish model loading or visual quality.
