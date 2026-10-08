"""Per-attempt evidence for the interactive Colab notebook (standard library only)."""
import json
import importlib.util
import os
import re
import signal
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


class NotebookRun:
    def __init__(self, content, project, settings):
        self.project = Path(project)
        root = Path(content) / "playworld-runs"
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
        self.path = Path(tempfile.mkdtemp(prefix=stamp, dir=root))
        self.scene = self.path / "scene"
        self.world = self.path / "world"
        self.source = self.path / "public_source"
        self.report = {"completed": False, "settings": settings, "stages": {}}
        self.save()

    def save(self):
        (self.path / "job.json").write_text(
            json.dumps(self.report, indent=2) + "\n", encoding="utf-8")

    def execute(self, command, stage):
        """Stream output and preserve failure evidence, including interrupted retries."""
        self.report["completed"] = False
        self.report["stages"][stage] = {"status": "running"}
        self.report.pop("backups", None)
        (self.path / "backup-index.json").unlink(missing_ok=True)
        self.save()
        started = time.monotonic()
        last_progress = started - 30
        pending_progress = None
        process = None
        try:
            with (self.path / f"{stage}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True,
                                           start_new_session=os.name == "posix")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    # tqdm uses carriage returns. Universal-newline decoding turns
                    # each update into a line; thousands of IOPub updates can stall
                    # Colab's output rendering. Keep the full disk log, but throttle
                    # progress in the notebook while relaying other messages.
                    if re.search(r"\d+%\|", line):
                        pending_progress = line
                        now = time.monotonic()
                        if now - last_progress < 30 and "100%|" not in line:
                            continue
                        last_progress = now
                        pending_progress = None
                    print(line, end="", flush=True)
                if pending_progress:
                    print(pending_progress, end="", flush=True)
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException as error:
            if process is not None and process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            # Exception messages/arguments may contain credentials. Record only the type.
            self.report["stages"][stage] = {
                "status": "failed", "error_type": type(error).__name__,
                "seconds": time.monotonic() - started,
            }
            self.save()
            raise
        self.report["stages"][stage] = {
            "status": "complete", "seconds": time.monotonic() - started,
        }
        if stage == "pipeline":
            if not (self.world / "world.json").is_file():
                self.report["stages"][stage]["status"] = "failed"
                self.save()
                raise RuntimeError("Pipeline exited without world.json")
            self.report["completed"] = True
        self.save()

    def prepare_backups(self):
        """Publish checkpoint chunks before making the world ZIP, before preview."""
        if not self.report["completed"]:
            raise RuntimeError("Finish reconstruction before preparing backups")
        final_step = self.report["settings"]["train_steps"] - 1
        checkpoint = self.scene / "gs" / "ckpts" / f"ckpt_{final_step}_rank0.pt"
        index = {"format": "playworld-backup-index/1", "complete": False, "artifacts": []}
        index_path = self.path / "backup-index.json"

        def publish():
            # Readers must never see half-written JSON while polling.
            temporary = index_path.with_suffix('.json.partial')
            temporary.write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')
            temporary.replace(index_path)
            self.report["backups"] = index
            self.save()

        publish()
        try:
            if not checkpoint.is_file():
                raise FileNotFoundError("Final checkpoint is missing")
            spec = importlib.util.spec_from_file_location(
                "playworld_colab_pack", Path(__file__).with_name("colab_pack.py"))
            packing = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(packing)
            root = Path(tempfile.mkdtemp(prefix="backup-", dir=self.path))
            for kind in ("final_checkpoint", "world_zip"):
                source = checkpoint if kind == "final_checkpoint" else self.bundle()
                directory = root / kind
                packing.pack(source, directory, step=final_step if kind == "final_checkpoint" else None)
                index["artifacts"].append({"kind": kind, "directory": directory.as_posix()})
                publish()
                print(f"Backup ready: {kind}", flush=True)
            index["complete"] = True
            publish()
        except BaseException as error:
            index["error_type"] = type(error).__name__
            publish()
            raise
        return index_path

    def bundle(self):
        """Fresh ZIP; failed attempts export logs, never an earlier world."""
        with tempfile.TemporaryDirectory(prefix="export-", dir=self.path) as temporary:
            staging = Path(temporary)
            shutil.copy(self.path / "job.json", staging / "job.json")
            for pattern in ("*.log", "scene/run-*", "scene/gpu-revisions.env",
                            "public_source/*.json", "public_source/EYEFULTOWER-LICENSE.txt"):
                for source in self.path.glob(pattern):
                    if source.is_file():
                        shutil.copy(source, staging / source.name)
            for source in (self.project / ".gpu").glob("*-freeze.txt"):
                shutil.copy(source, staging / source.name)
            if self.report["completed"]:
                shutil.copytree(self.world, staging / "world")
                # Preserve the camera frame for later quality comparisons. A
                # playable world alone cannot reconstruct its training views.
                # Copy only named metadata, rather than images or model weights.
                for relative in ("sparse/cameras.bin", "sparse/images.bin", "sparse/points3D.bin",
                                 "pose-diagnostics.json", "calibrated-source.json"):
                    source = self.scene / relative
                    if source.is_file():
                        target = staging / "scene" / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
            result = shutil.make_archive(str(self.path / "playworld-result"), "zip",
                                         root_dir=staging)
        return Path(result)
