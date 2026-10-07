import subprocess
from pathlib import Path

import pytest

from playworld import cli


@pytest.mark.parametrize('steps', [7000, 30000])
def test_interrupted_training_keeps_an_earlier_renderable_export(tmp_path, monkeypatch, steps):
    scene = tmp_path / 'scene'

    def interrupted_trainer(command, cwd):
        args = [str(arg) for arg in command]
        scaler = float(args[args.index('--steps-scaler') + 1])
        def scheduled(flag):
            start = args.index(flag) + 1
            values = []
            for value in args[start:]:
                if value.startswith('--'):
                    break
                values.append(int(int(value) * scaler))
            return values
        # Match the pinned trainer's adjusted save schedule. A process stopped
        # halfway through the run must already have saved both forms of output.
        completed = [step for step in scheduled('--save-steps') if 0 < step < steps // 2]
        renderable = [step for step in scheduled('--ply-steps') if step in completed]
        assert renderable, 'There is no saved reconstruction before interruption'
        output = scene / 'gs' / 'ply'
        output.mkdir(parents=True)
        (output / f'point_cloud_{renderable[-1] - 1}.ply').write_text('saved before interruption')
        raise subprocess.CalledProcessError(143, args)

    monkeypatch.setattr(cli, 'run', interrupted_trainer)
    with pytest.raises(subprocess.CalledProcessError):
        cli.train_splats(scene, tmp_path / 'gsplat', steps)
    assert cli.latest_ply(scene).read_text() == 'saved before interruption'
