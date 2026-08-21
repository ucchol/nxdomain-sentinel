#!/usr/bin/env python3
"""Train one or more family-holdout folds and checkpoint each.

    python scripts/train.py --folds 0 --epochs 6

Training and evaluation are separate steps on purpose. Training is the
expensive part, and keeping evaluation out of the same process means a failure
there costs seconds rather than the whole run. It also makes evaluation
re-runnable against a checkpoint without retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nxsentinel.data import splits as S  # noqa: E402
from nxsentinel.train.loop import TrainConfig, train_one_split  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
ARTIFACTS = REPO_ROOT / "artifacts"


# Domains such as nan.com, inf.com, na.com and null.com are real registrations.
# Pandas turns every one of them into a float under default NA handling, which
# then crashes the encoder. keep_default_na=False is required, not cosmetic.
_STR_COLS = {"core": str, "suffix": str, "family": str, "origin": str}


def load_dataset() -> pd.DataFrame:
    for name in ("domains.parquet", "domains.csv.gz"):
        p = PROCESSED / name
        if not p.exists():
            continue
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p, dtype=_STR_COLS, keep_default_na=False)
        # family, suffix and origin are low cardinality. Storing them as object
        # columns costs a Python string per row across 800k rows for no reason.
        for col in ("family", "suffix", "origin"):
            df[col] = df[col].astype("category")
        return df
    raise SystemExit("no dataset found, run scripts/build_dataset.py first")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_dataset()

    ARTIFACTS.mkdir(exist_ok=True)
    results = {}

    for fi in args.folds:
        # One fold at a time. Materialising all k up front copies the corpus k
        # times and is the difference between fitting in memory and not.
        (split,) = S.family_holdout_folds(df, k=args.k, seed=args.seed, only=[fi])
        print(f"\n=== fold {fi} ===", flush=True)
        print(f"holdout: {', '.join(split.meta['holdout_families'])}", flush=True)
        print(f"train rows {len(split.train):,}  val {len(split.val):,}", flush=True)

        cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            max_train_rows=args.max_train_rows,
        )
        t0 = time.time()
        model, ood, info = train_one_split(split, cfg)
        print(f"params {model.n_params():,}  trained in {time.time() - t0:.0f}s", flush=True)

        # Checkpoint before evaluation. Training is the expensive part and an
        # evaluation crash should never cost it.
        ckpt = ARTIFACTS / f"fold{fi}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": model.cfg.to_dict(),
                "ood": ood.to_dict(),
                "families": info["families"],
                "meta": split.meta,
            },
            ckpt,
        )
        print(f"checkpoint {ckpt} ({ckpt.stat().st_size / 1e6:.2f} MB)", flush=True)

        results[f"fold{fi}"] = {
            "history": info["history"],
            "n_params": model.n_params(),
            "train_seconds": info["seconds"],
            "holdout_families": split.meta["holdout_families"],
        }

    out = ARTIFACTS / "train_log.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(results)
    out.write_text(json.dumps(prev, indent=2, default=float))
    print(f"\nwrote {out}")
    print("now run: python scripts/evaluate.py --folds " + " ".join(map(str, args.folds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
