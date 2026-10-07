import json

import pytest

from playworld.cli import main


def test_failed_real_input_stage_leaves_an_honest_report(tmp_path):
    scene = tmp_path / "scene"
    with pytest.raises(FileNotFoundError):
        main(["all", str(tmp_path / "missing.mp4"), str(scene), "--out", str(tmp_path / "world"),
              "--vggt-dir", "unused", "--gsplat-dir", "unused"])
    report = json.loads((scene / "run.json").read_text())
    assert report["status"] == "failed"
    assert report["stages"]["frames"]["status"] == "failed"
    assert report["stages"]["frames"]["seconds"] >= 0
    assert "poses" not in report["stages"]
    assert report["source"]["sha256"] is None
