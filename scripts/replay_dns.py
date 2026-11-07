#!/usr/bin/env python3
"""Run the local, alert-only DNS replay on ordered JSONL."""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nxsentinel.ops.lexical import LexicalSignal  # noqa: E402
from nxsentinel.ops.replay import alert, events, windows  # noqa: E402
from nxsentinel.serve.predictor import Predictor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="timestamp-ordered DNS JSONL file")
    parser.add_argument("--model", type=Path, default=Path("artifacts/serving_model.pt"))
    parser.add_argument("--window-seconds", type=int, default=300)
    parser.add_argument("--min-distinct", type=int, default=5)
    parser.add_argument("--min-suspicious", type=int, default=2)
    parser.add_argument("--evidence-limit", type=int, default=10)
    parser.add_argument("--lexicon", type=Path, help="optional dictionary signal")
    parser.add_argument("--all-windows", action="store_true", help="emit every device window")
    args = parser.parse_args()
    if min(args.window_seconds, args.min_distinct, args.min_suspicious) < 1:
        parser.error("window length and count thresholds must be positive")
    if args.evidence_limit < 0:
        parser.error("evidence limit must be nonnegative")
    try:
        secret = bytes.fromhex(os.environ.get("NXS_REPLAY_HMAC_KEY", ""))
    except ValueError:
        parser.error("NXS_REPLAY_HMAC_KEY must be hexadecimal")
    if len(secret) < 32:
        parser.error("set NXS_REPLAY_HMAC_KEY to at least 32 random bytes encoded as hex")

    counts = Counter()
    started = time.perf_counter()
    try:
        predictor = Predictor.load(args.model, os.getenv("NXS_MODEL_SHA256"))
        lexical, lexical_version = LexicalSignal.load(args.lexicon) if args.lexicon else (None, None)
        with args.input.open(encoding="utf-8") as source:
            for row in windows(events(source, secret, counts), predictor.predict_batch,
                               seconds=args.window_seconds, lexical=lexical,
                               evidence_limit=args.evidence_limit):
                counts["windows"] += 1
                flagged = alert(row, args.min_distinct, args.min_suspicious)
                counts["alerts"] += flagged
                if args.all_windows or flagged:
                    record = {
                        **row, "alert": flagged, "model_version": predictor.version,
                        "experimental_novelty": not predictor.gate_is_trustworthy,
                        "lexical_version": lexical_version,
                        "model_sha256": predictor.artifact_sha256,
                        "all_windows": args.all_windows,
                    }
                    print(json.dumps(record, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        **counts, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)
    }, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
