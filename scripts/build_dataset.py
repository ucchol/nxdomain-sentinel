#!/usr/bin/env python3
"""Fetch the corpora, build the canonical dataset, and verify the splits.

    python scripts/build_dataset.py

Writes data/processed/domains.parquet and data/processed/dataset_report.json.
Safe to re-run. Raw checkouts are cached, so only the first run downloads.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nxsentinel.data import build as build_mod  # noqa: E402
from nxsentinel.data import sources, splits  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benign-limit", type=int, default=200_000)
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    print("fetching sources", flush=True)
    try:
        raw = sources.fetch_all(force=args.force_download)
    except ValueError as exc:
        ap.error(str(exc))
    for k, v in raw.items():
        print(f"  {k:32s} {v}")

    print("building canonical table", flush=True)
    df, report = build_mod.build(raw, benign_limit=args.benign_limit)
    report["sources"] = {source.key: source.revision for source in sources.ALL_SOURCES}
    report["tldextract_version"] = version("tldextract")

    print("checking cross-source isolation", flush=True)
    splits.assert_cross_source_clean(df)

    print(f"building {args.folds} family-holdout folds", flush=True)
    fold_summaries = []
    for fold in range(args.folds):
        (split,) = splits.family_holdout_folds(df, k=args.folds, seed=args.seed, only=[fold])
        fold_summaries.append(split.summary())
        del split
    report["splits"] = {
        "random": splits.random_split(df, seed=args.seed).summary(),
        "family_holdout": fold_summaries,
    }
    report["totals"] = {
        "rows": int(len(df)),
        "dga": int((df.label == 1).sum()),
        "benign": int((df.label == 0).sum()),
        "train_families": int(
            df[(df.origin == "train_corpus") & (df.label == 1)]["family"].nunique()
        ),
    }
    report["by_origin"] = {
        k: int(v) for k, v in df["origin"].value_counts().to_dict().items()
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_parquet = PROCESSED / "domains.parquet"
    try:
        df.to_parquet(out_parquet, index=False)
        written = out_parquet
    except Exception:
        # Fallback only. Parquet keeps dtypes; CSV needs the reader to disable
        # NA coercion or real domains like nan.com come back as floats.
        written = PROCESSED / "domains.csv.gz"
        df.to_csv(written, index=False, compression="gzip")

    (PROCESSED / "dataset_report.json").write_text(json.dumps(report, indent=2))

    print()
    print(f"wrote {written}")
    print(f"wrote {PROCESSED / 'dataset_report.json'}")
    print()
    t = report["totals"]
    print(
        f"rows {t['rows']:,}  dga {t['dga']:,}  benign {t['benign']:,}  "
        f"train families {t['train_families']}"
    )
    print(f"unseen cross-source families {report['n_unseen_families']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
