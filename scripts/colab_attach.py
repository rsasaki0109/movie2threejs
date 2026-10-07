"""Register one existing Colab assignment with the official CLI; never provision a VM.

Run with the Python interpreter in the google-colab-cli installation environment.
The CLI's StateStore API retains proxy credentials without printing them.
"""
import argparse
from colab_cli.common import state
from colab_cli.state import SessionState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--name", default="playworld-l4")
    args = parser.parse_args()
    matches = [a for a in state.client.list_assignments() if a.endpoint == args.endpoint]
    if len(matches) != 1:
        raise RuntimeError("The specified existing Colab assignment is not currently available")
    assignment = matches[0]
    existing = state.store.get(args.name)
    if existing and existing.endpoint != args.endpoint:
        raise RuntimeError("The requested local session name already belongs to another runtime")
    if existing:
        print(f"Already attached: {args.name} ({assignment.accelerator.name})")
        return
    proxy = assignment.runtime_proxy_info
    state.store.add(SessionState(
        name=args.name, endpoint=assignment.endpoint, token=proxy.token, url=proxy.url,
        token_expires_at=proxy.expires_at(), variant=assignment.variant.name,
        accelerator=assignment.accelerator.name, machine_shape=assignment.machine_shape.name,
    ))
    print(f"Attached existing runtime: {args.name} ({assignment.accelerator.name})")


if __name__ == "__main__":
    main()
