# Colab CLI (remote GPU execution not yet validated)

The official [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)
supports Linux/macOS. On this PC it is installed as version 0.7.4 in WSL Ubuntu-22.04,
at `/root/.local/bin/colab`. Google authentication is still required. `--help`,
`version`, `sessions --help`, `exec --help` and `upload --help` ran successfully;
`sessions` reached the initial OAuth authorization flow. No runtime was allocated.

Authenticate in your own PowerShell terminal:

```powershell
wsl -d Ubuntu-22.04 --exec /root/.local/bin/colab --auth oauth2 sessions
```

Open the URL printed by the CLI, approve in Google, and paste the authorization code
back into that terminal. Do not put the code or an HF token in chat or source files.
Then the agent can inspect existing sessions and account availability. Creating a new
runtime can consume compute units; these examples do not allocate one.

In WSL, from the project directory, use an existing suitable session (`SESSION`):

```bash
/root/.local/bin/colab exec -s SESSION -f scripts/colab_job.py
/root/.local/bin/colab upload -s SESSION .cache/playworld-colab.zip /content/playworld-colab.zip
/root/.local/bin/colab exec -s SESSION --timeout 7200 --env PLAYWORLD_CLI_ACTION=run -f scripts/colab_job.py
/root/.local/bin/colab download -s SESSION /content/playworld-result.zip .cache/playworld-result.zip
```

The first command checks actual GPU hardware and approved SAM 3 access, without setup
or inference. It reads `HF_TOKEN` from the existing remote environment or Colab Secrets;
if the secret is unavailable in this runtime, the preflight reports that and the run
does not proceed. SAM 3 uses the same L4/A100/native-bf16 requirement as `run.sh`.

The explicit `run` action unpacks the project, runs the shared setup and pipeline scripts,
and packages the world and measured evidence. On setup/pipeline failure it packages logs
without exporting a stale world. The wrapper never provisions or stops a runtime. Preserve
the result before rerunning, and release a task-owned runtime when the work finishes.

The existing browser-upload notebook remains available; its `files.upload()` cells should
not be executed through the CLI. The CLI entry point avoids those interactive cells.
