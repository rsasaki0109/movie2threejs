import json

import numpy as np
import pytest

from playworld import cli
from playworld.colmap_io import Reconstruction, write_model


@pytest.mark.parametrize("points", [np.empty((0, 3)), np.full((4, 3), np.nan)])
def test_invalid_vggt_export_fails_pose_stage_before_training(tmp_path, monkeypatch, points):
    scene = tmp_path / "scene"
    video = tmp_path / "input.mp4"
    video.write_bytes(b"not decoded when reusing extracted frames")
    commands = []

    def fake_vggt(command, cwd=None):
        commands.append(command)
        write_model(Reconstruction({}, {}, points, np.zeros((len(points), 3), dtype=np.uint8)), scene / "sparse")

    monkeypatch.setattr(cli, "run", fake_vggt)
    with pytest.raises(ValueError, match="four finite points"):
        cli.main(["all", str(video), str(scene), "--out", str(tmp_path / "world"),
                  "--vggt-dir", "unused", "--gsplat-dir", "unused", "--start-at", "poses",
                  "--pose-confidence", "1.5"])
    assert len(commands) == 1
    assert "--conf_thres_value=1.5" in commands[0]
    report = json.loads((scene / "run.json").read_text())
    assert report["status"] == "failed"
    assert report["stages"]["poses"]["status"] == "failed"
    assert "train" not in report["stages"]
