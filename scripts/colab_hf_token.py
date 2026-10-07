"""Transfer the locally authenticated HF token to an existing Colab runtime.

Run in WSL after `uvx --from huggingface-hub hf auth login`. The token is never
printed, passed as a command-line argument, or included in project archives.
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--token-file", type=Path, help="explicit existing HF authentication cache")
    args = parser.parse_args()
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    token_path = args.token_file or Path(os.environ.get("HF_TOKEN_PATH", str(hf_home / "token")))
    token = os.environ.get("HF_TOKEN") or (token_path.read_text().strip() if token_path.is_file() else None)
    if not token:
        raise RuntimeError("Authenticate with hf auth login first; no HF token is available")
    with tempfile.NamedTemporaryFile(mode="w", prefix="playworld-hf-", delete=False) as secret:
        secret_path = Path(secret.name)
        secret_path.chmod(0o600)
        secret.write(token)
    try:
        subprocess.run([str(Path.home() / ".local/bin/colab"), "upload", "-s", args.session,
                        str(secret_path), "/content/.playworld-hf-token"], check=True)
    finally:
        secret_path.unlink(missing_ok=True)
    print("HF credential transferred; the next preflight consumes and removes the remote transfer file.")


if __name__ == "__main__":
    main()
