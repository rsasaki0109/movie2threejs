import importlib.util
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def executor(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('colab_execute_test', scripts/'colab_execute.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path/'code.py'
    source.write_text('print(123)')
    monkeypatch.setattr(module, 'runtime', lambda *args: {'url':'https://example.prod.colab.dev', 'token':'fixture-secret'})
    clients, executed = [], []
    kernels = [{'id':'other','execution_state':'busy'}, {'id':'chosen','execution_state':'idle'}]

    class Client:
        def __init__(self, kernel_id=None, **options):
            self.kernel_id = kernel_id
            self.started = False
            self.stopped = False
            clients.append(self)
        def list_kernels(self):
            return kernels
        def start(self):
            self.started = True
        def execute_interactive(self, code, **options):
            executed.append((self.kernel_id, code))
            return {'content':{'status':'ok'}}
        def stop(self):
            assert not self._own_kernel
            self.stopped = True

    monkeypatch.setitem(sys.modules, 'jupyter_kernel_client', types.SimpleNamespace(
        KernelClient=Client, JupyterSubprotocol=types.SimpleNamespace(DEFAULT='test')))
    def invoke(selection):
        monkeypatch.setattr(sys, 'argv', ['colab_execute','--session','owned-runtime','--file',str(source),*selection])
        return module.main()
    return invoke, kernels, clients, executed


def test_explicit_observed_kernel_runs_without_using_other_busy_kernel(executor, capfd):
    invoke, _, clients, executed = executor
    assert invoke(['--kernel-id','chosen']) == 0
    assert executed == [('chosen','print(123)')]
    assert all(c.kernel_id == 'chosen' for c in clients)
    assert clients[-1].stopped
    assert 'fixture-secret' not in ''.join(capfd.readouterr())


@pytest.mark.parametrize('selection,state', [('missing','idle'), ('chosen','busy')])
def test_missing_or_busy_requested_kernel_never_executes(executor, capfd, selection, state):
    invoke, kernels, clients, executed = executor
    kernels[1]['execution_state'] = state
    assert invoke(['--kernel-id',selection]) == 1
    assert executed == [] and not any(c.started for c in clients)
    assert 'fixture-secret' not in ''.join(capfd.readouterr())


def test_multiple_kernels_require_explicit_selection(executor, capfd):
    invoke, _, clients, executed = executor
    assert invoke([]) == 1
    assert executed == [] and not any(c.started for c in clients)
