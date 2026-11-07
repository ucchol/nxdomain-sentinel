"""Replay timestamp-ordered DNS JSONL and score device windows locally."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator

NXDOMAIN = 3


@dataclass(frozen=True)
class Event:
    ts: float
    device_id: str
    query: str
    nxdomain: bool


def device_id(secret: bytes, client: str) -> str:
    return hmac.new(secret, client.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def events(
    lines: Iterable[str], secret: bytes, counts: Counter
) -> Iterator[Event]:
    """Accept Zeek dns.log JSON or rows with a normalized `client` field."""
    last_ts = -1.0
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected a JSON object")
            client = row.get("client", row.get("id.orig_h"))
            query = row.get("query")
            rcode = row.get("rcode")
            if query is None or rcode is None:
                counts["missing_query_or_rcode"] += 1
                continue
            ts = float(row["ts"])
            if not math.isfinite(ts) or ts < 0:
                raise ValueError("timestamp must be finite and nonnegative")
            if ts < last_ts:
                raise ValueError("timestamps must be in nondecreasing order")
            if not isinstance(client, str) or not client.strip():
                raise ValueError("missing client identifier")
            if not isinstance(query, str) or not 1 <= len(query) <= 254:
                raise ValueError("invalid query length or type")
            query = query.lower().rstrip(".")
            if type(rcode) is not int or not 0 <= rcode <= 4095:
                raise ValueError("rcode must be a DNS response code")
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc
        last_ts = ts
        counts["responses"] += 1
        counts["nxdomain_responses"] += rcode == NXDOMAIN
        yield Event(ts, device_id(secret, client), query, rcode == NXDOMAIN)


def _score_window(groups, start, seconds, predict, batch_size, evidence_limit):
    names = sorted({query for _, failed in groups.values() for query in failed})
    scores = {}
    for offset in range(0, len(names), batch_size):
        batch = names[offset : offset + batch_size]
        results = predict(batch)
        scores.update(zip(batch, results, strict=True))

    for host, (total, failed) in sorted(groups.items()):
        dga = []
        novel = []
        combined = []
        unscored = 0
        for query in sorted(failed):
            verdict = scores[query]
            if verdict.verdict == "unparseable":
                unscored += 1
                continue
            if verdict.verdict == "dga":
                dga.append(query)
            if verdict.above_novelty_threshold:
                novel.append(query)
            if verdict.verdict == "dga" or verdict.above_novelty_threshold:
                combined.append(query)
        case_id = f"{host}-{start}-{seconds}"
        yield {
            "case_id": case_id,
            "device_id": host,
            "window_start": start,
            "window_end": start + seconds,
            "responses": total,
            "nxdomain_responses": sum(failed.values()),
            "distinct_nxdomain": len(failed),
            "unscored_names": unscored,
            "dga_names": len(dga),
            "novelty_names": len(novel),
            "combined_names": len(combined),
            "evidence": [
                {
                    "query": query,
                    "count": failed[query],
                    "dga_score": scores[query].dga_probability,
                    "novelty_score": scores[query].novelty_score,
                    "above_dga_threshold": scores[query].verdict == "dga",
                    "above_novelty_threshold": scores[query].above_novelty_threshold,
                }
                for query in combined[:evidence_limit]
            ],
        }


def windows(
    records: Iterable[Event],
    predict,
    seconds: int = 300,
    batch_size: int = 500,
    evidence_limit: int = 10,
) -> Iterator[dict]:
    """Score each distinct failed name once per window, preserving host counts."""
    if seconds < 1 or batch_size < 1 or evidence_limit < 0:
        raise ValueError("window and batch sizes must be positive, evidence limit nonnegative")
    start = None
    groups: dict[str, tuple[int, Counter]] = {}
    for event in records:
        bucket = int(event.ts // seconds) * seconds
        if start is not None and bucket != start:
            yield from _score_window(groups, start, seconds, predict, batch_size, evidence_limit)
            groups.clear()
        start = bucket
        total, failed = groups.get(event.device_id, (0, Counter()))
        if event.nxdomain:
            failed[event.query] += 1
        groups[event.device_id] = total + 1, failed
    if start is not None:
        yield from _score_window(groups, start, seconds, predict, batch_size, evidence_limit)


def alert(row: dict, min_distinct: int, min_suspicious: int) -> bool:
    return row["distinct_nxdomain"] >= min_distinct and row["combined_names"] >= min_suspicious
