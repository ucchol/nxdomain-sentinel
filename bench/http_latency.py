#!/usr/bin/env python3
"""Measure batch HTTP latency including JSON and network time."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import httpx
import numpy as np

from latency import sample_domains


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="base URL for a running Sentinel service")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 500 or min(args.requests, args.concurrency) < 1:
        parser.error("batch size must be 1–500 and requests and concurrency must be positive")
    token = os.environ.get("NXS_BENCH_TOKEN")
    if token and urlsplit(args.url).scheme != "https":
        parser.error("an authentication token requires HTTPS")

    url = args.url.rstrip("/") + "/v1/score/batch"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(headers=headers, timeout=30, trust_env=False) as client:
        def run_one(index):
            domains = sample_domains(args.batch_size, seed=index)
            start = time.perf_counter()
            response = client.post(url, json={"domains": domains})
            response.raise_for_status()
            result = response.json()
            if result["count"] != args.batch_size:
                raise ValueError("unexpected response count")
            return (
                (time.perf_counter() - start) * 1000,
                (result["model_version"], result.get("model_sha256")),
            )

        try:
            first_ms, version = run_one(0)
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                samples = list(pool.map(run_one, range(1, args.requests + 1)))
            elapsed = time.perf_counter() - started
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            parser.error(str(exc))

    if any(sample_version != version for _, sample_version in samples):
        parser.error("model version changed during the benchmark")
    latencies = [latency for latency, _ in samples]
    print(json.dumps({
        "model_version": version[0],
        "model_sha256": version[1],
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "requests_after_first": args.requests,
        "first_request_ms": round(first_ms, 3),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "domains_per_second": round(args.requests * args.batch_size / elapsed, 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
