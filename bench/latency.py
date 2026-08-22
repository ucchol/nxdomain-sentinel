#!/usr/bin/env python3
"""Measure inference latency for the exported serving model.

    python bench/latency.py

Reports p50, p95 and p99 across batch sizes. Batch size matters more than
anything else here: per-domain cost falls sharply with batching, which is the
whole argument for exposing a batch endpoint rather than telling callers to loop.

These are in-process inference numbers, so they exclude HTTP overhead, JSON
serialisation, and network time. End-to-end numbers come from load testing the
deployed service and are reported separately. Conflating the two is how latency
claims end up being quietly wrong.
"""

from __future__ import annotations

import json
import platform
import random
import string
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nxsentinel.serve.predictor import Predictor  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"
BATCH_SIZES = (1, 10, 50, 100, 500)


def sample_domains(n: int, seed: int = 0) -> list[str]:
    """Mix of realistic benign names and random-looking labels."""
    rng = random.Random(seed)
    benign = ["github.com", "wikipedia.org", "cloudflare.com", "nytimes.com"]
    out = []
    for i in range(n):
        if i % 3 == 0:
            out.append(rng.choice(benign))
        else:
            length = rng.randint(8, 24)
            label = "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(length))
            out.append(f"{label}.{rng.choice(['com', 'net', 'info', 'biz'])}")
    return out


def main() -> int:
    model_path = ARTIFACTS / "serving_model.pt"
    if not model_path.exists():
        raise SystemExit("no serving model, run scripts/export_model.py first")

    p = Predictor.load(model_path)

    print(f"model      {p.version}")
    print(f"params     {sum(x.numel() for x in p.model.parameters()):,}")
    print(f"artifact   {model_path.stat().st_size / 1e6:.2f} MB")
    print(f"machine    {platform.processor() or platform.machine()}\n")

    print(f"  {'batch':>7}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}{'per domain':>13}")
    print(f"  {'-' * 50}")

    results = []
    for bs in BATCH_SIZES:
        r = p.benchmark(sample_domains(bs), repeats=30 if bs < 500 else 10)
        results.append(r)
        print(
            f"  {bs:>7}{r['p50_ms']:>10.2f}{r['p95_ms']:>10.2f}"
            f"{r['p99_ms']:>10.2f}{r['per_domain_p50_ms']:>12.3f}ms"
        )

    single = results[0]["per_domain_p50_ms"]
    best = min(r["per_domain_p50_ms"] for r in results)
    print(f"\n  batching gives a {single / best:.1f}x reduction in per-domain cost")

    out = ARTIFACTS / "latency.json"
    out.write_text(json.dumps({"results": results, "model": p.version}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
