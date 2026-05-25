import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SUMMARY_ARGS = [
    "--print-only",
    "--csv",
    "twoWiseExp_counts_B400.csv",
    "--N",
    "16000057",
    "--B",
    "400",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run the paper experiments with either the Python or C++ implementation.",
    )
    parser.add_argument(
        "implementation",
        nargs="?",
        choices=("python", "cpp"),
        default="python",
        help="Implementation backend to run.",
    )
    args, backend_args = parser.parse_known_args()

    if not backend_args:
        backend_args = DEFAULT_SUMMARY_ARGS

    runner = ROOT / args.implementation / "run_experiment.py"
    command = [sys.executable, str(runner), *backend_args]
    raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
