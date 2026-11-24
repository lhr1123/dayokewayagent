"""CLI for a synthetic demo or a batch of Task 1 samples."""

import argparse
from io import TextIOWrapper
from pathlib import Path
import sys
from zipfile import BadZipFile

from .runner import run_batch, run_demo


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="run a known-answer synthetic case")
    demo.add_argument("--case", choices=("segment", "region"), default="segment")
    demo.add_argument("--offline", action="store_true", help="use a scripted model without API calls")

    run = commands.add_parser("run", help="run Task 1 with the configured model endpoint")
    run.add_argument("--input", type=Path, required=True, help="data ZIP, directory or Task 1 question file")
    run.add_argument("--output", type=Path, required=True, help="Task 1 result JSONL")
    run.add_argument("--trace", type=Path, help="optional evidence and timing JSONL")
    run.add_argument("--sample-id", action="append", help="optional sample filter; repeat for more IDs")
    args = parser.parse_args(argv)

    try:
        if args.command == "demo":
            return run_demo(args.case, offline=args.offline)
        return run_batch(args.input, args.output, trace_path=args.trace, sample_ids=args.sample_id)
    except (ValueError, OSError, BadZipFile) as exc:
        print(f"Task 1: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
