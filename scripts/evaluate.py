#!/usr/bin/env python3
"""Score trained fold checkpoints.

    python scripts/evaluate.py --folds 0 1 2 3 4

Loads each checkpoint written by scripts/train.py and reports the contrast the
project exists to show, one model scored on families it trained on and on
families it has never seen, plus a cross-source set from a different author.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nxsentinel.data import splits as S  # noqa: E402
from nxsentinel.model.charcnn import CharCNN, CharCNNConfig  # noqa: E402
from nxsentinel.model.ood import MahalanobisOOD  # noqa: E402
from nxsentinel.train.evaluate import evaluate_fold  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
ARTIFACTS = REPO_ROOT / "artifacts"

_STR_COLS = {"core": str, "suffix": str, "family": str, "origin": str}


def load_dataset() -> pd.DataFrame:
    for name in ("domains.parquet", "domains.csv.gz"):
        p = PROCESSED / name
        if not p.exists():
            continue
        df = (
            pd.read_parquet(p)
            if p.suffix == ".parquet"
            else pd.read_csv(p, dtype=_STR_COLS, keep_default_na=False)
        )
        for col in ("family", "suffix", "origin"):
            df[col] = df[col].astype("category")
        return df
    raise SystemExit("no dataset found, run scripts/build_dataset.py first")


def load_checkpoint(path: Path):
    ck = torch.load(path, weights_only=True, map_location="cpu")
    model = CharCNN(CharCNNConfig.from_dict(ck["config"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, MahalanobisOOD.from_dict(ck["ood"]), ck


def report(fi: int, res: dict) -> None:
    tu = res["test_unseen"]
    print(f"\n  {'partition':<18}{'AUC':>9}{'AUPRC':>9}{'recall@0.1%FPR':>17}")
    print(f"  {'-' * 53}")
    for label, key in (
        ("seen families", "test_seen"),
        ("UNSEEN families", "test_unseen"),
        ("cross-source", "cross_source"),
    ):
        r = res[key]
        print(
            f"  {label:<18}{r['auc']:>9.4f}{r['auprc']:>9.4f}"
            f"{r['recall@fpr0.1%']:>17.4f}"
        )

    o = res["ood"]
    print(f"\n  OOD gate AUC, unseen vs seen families: {o['auc']:.4f}")
    print(
        f"  unseen families flagged novel at 5% seen-family FPR: "
        f"{o['unseen_flagged_at_5pct_seen_fpr']:.4f}"
    )

    print("\n  per-family recall on held-out families (worst first):")
    for fam, m in sorted(tu["per_family"].items(), key=lambda kv: kv[1]["recall"]):
        print(f"    {fam:<18} n={m['n']:>6,}  recall={m['recall']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_dataset()
    cross = S.cross_source_sets(df)

    out = ARTIFACTS / "fold_results.json"
    results = json.loads(out.read_text()) if out.exists() else {}

    for fi in args.folds:
        ckpt = ARTIFACTS / f"fold{fi}.pt"
        if not ckpt.exists():
            print(f"fold {fi}: no checkpoint at {ckpt}, skipping", flush=True)
            continue

        (split,) = S.family_holdout_folds(df, k=args.k, seed=args.seed, only=[fi])
        model, ood, ck = load_checkpoint(ckpt)

        print(f"\n=== fold {fi} ===", flush=True)
        print(f"holdout: {', '.join(split.meta['holdout_families'])}", flush=True)

        res = evaluate_fold(model, ood, split, cross)
        res["meta"] = {
            "fold": fi,
            "holdout_families": split.meta["holdout_families"],
            "n_params": model.n_params(),
        }
        results[f"fold{fi}"] = res
        report(fi, res)

        del split, model, ood

    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
