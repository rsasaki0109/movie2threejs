import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def preflight_module(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "colab_job", Path(__file__).parents[1] / "scripts/colab_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CONTENT = tmp_path
    gpu = types.SimpleNamespace(name="Test L4", major=8, total_memory=22 * 2**30)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, get_device_properties=lambda index: gpu)))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("PLAYWORLD_PROMPTS", "box")
    return module


def test_secrets_token_trims_clipboard_newlines(preflight_module, monkeypatch, capsys):
    module = preflight_module
    secret = "test-read-credential"
    colab = types.ModuleType("google.colab")
    requested = []

    def get(name):
        requested.append(name)
        return " " + secret + "\r\n"

    colab.userdata = types.SimpleNamespace(get=get)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    headers = []

    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def open_request(request, timeout):
        headers.append(request.get_header("Authorization"))
        return Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", open_request)
    assert module.preflight()
    assert requested == ["HF_TOKEN"]
    assert headers == ["Bearer " + secret]
    assert module.os.environ["HF_TOKEN"] == secret
    report = (module.CONTENT / "playworld-preflight.json").read_text()
    assert json.loads(report)["sam3_access"] == "approved"
    assert secret not in capsys.readouterr().out + report


@pytest.mark.parametrize("suffix", ["\nextra", "\x00", "é", " extra"])
def test_invalid_token_never_reaches_headers(preflight_module, monkeypatch, capsys, suffix):
    module = preflight_module
    secret = "test-private-credential" + suffix
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=lambda name: secret)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))

    def forbidden_request(*args, **kwargs):
        pytest.fail("Malformed credential must never be passed to HTTP")

    monkeypatch.setattr(module.urllib.request, "urlopen", forbidden_request)
    assert not module.preflight()
    report = (module.CONTENT / "playworld-preflight.json").read_text()
    assert json.loads(report)["sam3_access"] == "HF_TOKEN format invalid"
    assert "test-private-credential" not in capsys.readouterr().out + report
