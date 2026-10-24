#!/usr/bin/env python3
"""Load a saved SHG index and profile its searches."""

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("sift", "enron"), default="sift")
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=500)
    for option in ("index-file", "query-file", "groundtruth-file", "output",
                   "performance-output", "layer-averages-output"):
        parser.add_argument(f"--{option}")
    parser.add_argument("--executable", default=str(ROOT / "build/shg-profile/shg_profile"))
    args = parser.parse_args()
    command = [args.executable, "--search-only", "--dataset", args.dataset,
               "--M", str(args.M), "--k", str(args.k),
               "--ef-search", str(args.ef_search), "--warmup", str(args.warmup)]
    for option in ("index-file", "query-file", "groundtruth-file", "output",
                   "performance-output", "layer-averages-output"):
        value = getattr(args, option.replace("-", "_"))
        if value:
            command.extend((f"--{option}", value))
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
