#!/usr/bin/env python3
"""Detect the local GPU's compute capability, sync .clangd, then run `make ARCH=<sm_xx>`."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLANGD_PATH = ROOT / ".clangd"


def detect_arch() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        sys.exit(f"error: could not query GPU via nvidia-smi ({e}); pass ARCH=sm_xx to make directly")

    caps = sorted({line.strip() for line in out.splitlines() if line.strip()})
    if not caps:
        sys.exit("error: nvidia-smi returned no GPUs")
    if len(caps) > 1:
        print(f"warning: multiple GPU compute capabilities detected {caps}, using {caps[0]}", file=sys.stderr)

    major, minor = caps[0].split(".")
    return f"sm_{major}{minor}"


def sync_clangd(arch: str) -> None:
    text = CLANGD_PATH.read_text()
    new_text = re.sub(r"--cuda-gpu-arch=sm_\d+", f"--cuda-gpu-arch={arch}", text)
    if new_text != text:
        CLANGD_PATH.write_text(new_text)
        print(f"synced .clangd -> {arch}")


def main() -> None:
    arch = detect_arch()
    sync_clangd(arch)
    cmd = ["make", f"ARCH={arch}", *sys.argv[1:]]
    print(f"+ {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
