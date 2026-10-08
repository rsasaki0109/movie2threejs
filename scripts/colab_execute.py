"""Execute code in an existing CLI Colab session with native Windows networking.

Requires jupyter-kernel-client==0.9.0. Does not allocate or stop runtimes.
The authenticated official CLI session must already exist in WSL.
"""
import argparse
import json
import logging
from pathlib import Path
import sys

from colab_download import runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--file', required=True, type=Path)
    parser.add_argument('--cells', type=int, help='Execute only the first N notebook code cells')
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--distro', default='Ubuntu-22.04')
    parser.add_argument('--sdk-python', default='/root/.local/share/uv/tools/google-colab-cli/bin/python')
    args = parser.parse_args()
    if args.cells is not None and (args.cells < 1 or args.file.suffix != '.ipynb'):
        parser.error('--cells requires a notebook and a positive cell count')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    text = args.file.read_text(encoding='utf-8-sig')
    if args.file.suffix == '.ipynb':
        cells = [c['source'] for c in json.loads(text)['cells'] if c['cell_type'] == 'code']
        cells = cells[:args.cells] if args.cells is not None else cells
        blocks = [''.join(c) if isinstance(c, list) else c for c in cells]
    else:
        blocks = [text]
    # Windows consoles may reject tqdm's block characters. This also makes
    # redirected output deterministic across the local code page.
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    # SDK exception logs can contain credential-bearing proxy URLs.
    logging.disable(logging.CRITICAL)
    from jupyter_kernel_client import KernelClient, JupyterSubprotocol
    info = runtime(args.session, args.sdk_python, args.distro)
    token = info['token']
    client = None
    try:
        options = dict(
            server_url=info['url'], token=token,
            client_kwargs={'subprotocol': JupyterSubprotocol.DEFAULT,
                           'extra_params': {'colab-runtime-proxy-token': token}},
            headers={'X-Colab-Client-Agent': 'colab-cli',
                     'X-Colab-Runtime-Proxy-Token': token})
        client = KernelClient(**options)
        # Closing a client must preserve the already provisioned runtime/kernel.
        client._own_kernel = False
        kernels = client.list_kernels()
        if len(kernels) > 1:
            print('Multiple kernels; select the intended one with the official CLI.', file=sys.stderr)
            return 1
        if kernels:
            if kernels[0].get('execution_state') == 'busy':
                print('Existing kernel is busy; inspect its job/log before retrying.', file=sys.stderr)
                return 1
            client = KernelClient(kernel_id=kernels[0]['id'], **options)
            client._own_kernel = False
        client.start()
        def output(message):
            content = message.get('content', {})
            kind = message.get('header', {}).get('msg_type')
            if kind == 'stream':
                print(content.get('text', '').replace(token, '[redacted]'), end='', flush=True)
            elif kind == 'error':
                print('Remote error type: ' + content.get('ename', 'Error'), flush=True)
        for index, code in enumerate(blocks):
            print(f'Executing code block {index + 1}/{len(blocks)}', flush=True)
            reply = client.execute_interactive(code, output_hook=output, timeout=args.timeout)
            if reply['content']['status'] != 'ok':
                raise RuntimeError('Remote execution did not complete')
    except Exception as error:
        print('Execution failed: ' + type(error).__name__ +
              '; the remote kernel may still be running. Check its job/log before retrying.',
              file=sys.stderr, flush=True)
        return 1
    finally:
        if client is not None:
            try:
                client.stop()
            except Exception:
                pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
