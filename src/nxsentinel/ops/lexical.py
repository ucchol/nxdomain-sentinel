"""Dictionary-DGA candidate signal using benign word-pair counts."""

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path


class LexicalSignal:
    def __init__(self, words: Iterable[str], pairs: dict[str, int] | None = None):
        self.words = {w for w in words if len(w) >= 3 and w.isascii() and w.isalpha()}
        if not self.words:
            raise ValueError("the lexicon needs ASCII words of at least three letters")
        self.lengths = sorted({len(word) for word in self.words}, reverse=True)
        self.pairs = pairs if pairs is not None else {}

    def segment(self, core: str) -> tuple[str, ...]:
        label = core.rsplit(".", 1)[-1]
        if label in self.words or not (label.isascii() and label.isalpha()):
            return ()
        paths: dict[int, tuple[str, ...]] = {len(label): ()}
        for start in range(len(label) - 3, -1, -1):
            best = None
            for length in self.lengths:
                end = start + length
                word = label[start:end]
                if end in paths and word in self.words:
                    candidate = (word, *paths[end])
                    if best is None or len(candidate) < len(best):
                        best = candidate
            if best is not None:
                paths[start] = best
        return paths.get(0, ())

    @classmethod
    def fit(cls, words: Iterable[str], benign_cores: Iterable[str]):
        signal = cls(words)
        pairs = Counter()
        for core in benign_cores:
            pairs.update(" ".join(pair) for pair in pairwise(signal.segment(core)))
        if not pairs:
            raise ValueError("no word pairs found in the benign reference")
        signal.pairs = dict(pairs)
        return signal

    def flags(self, core: str) -> bool:
        return any(" ".join(pair) not in self.pairs for pair in pairwise(self.segment(core)))

    def save(self, path: Path) -> None:
        payload = {"schema": 1, "words": sorted(self.words), "pairs": self.pairs}
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload["schema"] != 1:
            raise ValueError("unsupported lexicon schema")
        return cls(payload["words"], payload["pairs"]), hashlib.sha256(raw).hexdigest()
