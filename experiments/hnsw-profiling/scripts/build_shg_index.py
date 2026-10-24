#!/usr/bin/env python3
"""Build and save an SHG index for SIFT or Enron."""

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("sift", "enron"), default="sift")
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=128)
    parser.add_argument("--data-file")
    parser.add_argument("--index-file")
    parser.add_argument("--performance-output")
    parser.add_argument("--executable", default=str(ROOT / "build/shg-profile/shg_profile"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    command = [args.executable, "--build-only", "--dataset", args.dataset,
               "--M", str(args.M), "--ef-construction", str(args.ef_construction)]
    for option in ("data-file", "index-file", "performance-output"):
        value = getattr(args, option.replace("-", "_"))
        if value:
            command.extend((f"--{option}", value))
    if args.force:
        command.append("--force")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
