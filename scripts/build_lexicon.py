#!/usr/bin/env python3
"""Build a dictionary signal from a fixed word list and earlier benign names."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nxsentinel.data.normalize import normalize  # noqa: E402
from nxsentinel.ops.lexical import LexicalSignal  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("word_list", type=Path, help="one English word per line")
    parser.add_argument("benign_domains", type=Path, help="one earlier benign DNS name per line")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        words = args.word_list.read_text(encoding="utf-8").lower().splitlines()
        with args.benign_domains.open(encoding="utf-8") as domains:
            cores = (normalize(domain) for domain in domains)
            signal = LexicalSignal.fit((word.strip() for word in words),
                                       (entry.core for entry in cores if entry.ok))
        signal.save(args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{len(signal.words)} words, {len(signal.pairs)} observed word pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
