#!/usr/bin/env python3
"""Compare DNS replay methods on forward-in-time device windows."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nxsentinel.ops.benchmark import compare  # noqa: E402


def read_rows(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="replay output from --all-windows")
    parser.add_argument("labels", type=Path, help="reviewed case labels in JSONL")
    parser.add_argument("--test-features", type=Path, help="optional separate test network")
    parser.add_argument("--calibration-end", type=int, required=True, help="Unix seconds")
    parser.add_argument("--test-start", type=int, required=True, help="Unix seconds")
    parser.add_argument("--alerts-per-1000-device-days", type=float, required=True)
    parser.add_argument("--min-distinct", type=int, default=5)
    args = parser.parse_args()
    try:
        calibration = list(read_rows(args.features))
        test = list(read_rows(args.test_features)) if args.test_features else calibration
        result = compare(
            calibration, test, read_rows(args.labels), args.calibration_end, args.test_start,
            args.alerts_per_1000_device_days, args.min_distinct,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
