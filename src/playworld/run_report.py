"""Persist measured stage times and provenance, including interrupted/failed runs."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class RunReport:
    def __init__(self, path: Path, video: Path, settings: dict):
        self.path = path
        self.started = time.perf_counter()
        try:
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            gpu = None
        digest = hashlib.sha256()
        if video.is_file():
            with video.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
        self.data = {
            "format": "playworld-run/1", "started_utc": utc_now(), "status": "running",
            "source": {"kind": "video", "name": video.name,
                       "sha256": digest.hexdigest() if video.is_file() else None,
                       "bytes": video.stat().st_size if video.is_file() else None},
            "platform": platform.platform(), "python": platform.python_version(),
            "gpu": gpu, "settings": settings, "stages": {},
        }
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def stage(self, name, fn, *args):
        started = time.perf_counter()
        record = {"status": "running", "started_utc": utc_now()}
        self.data["stages"][name] = record
        self.save()
        try:
            result = fn(*args)
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            self.data["status"] = "failed"
            raise
        else:
            record["status"] = "ok"
            return result
        finally:
            record["seconds"] = round(time.perf_counter() - started, 3)
            record["finished_utc"] = utc_now()
            self.save()
            print(f"[{name}] {record['status']} {record['seconds']} s", flush=True)

    def finish(self, world):
        self.data.update(status="ok", finished_utc=utc_now(),
                         total_seconds=round(time.perf_counter() - self.started, 3),
                         stats=world["stats"])
        self.save()
