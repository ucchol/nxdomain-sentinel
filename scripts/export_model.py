#!/usr/bin/env python3
"""Package a trained fold into a single self-contained serving artifact.

    python scripts/export_model.py --fold 0

The point of this script is that a threshold separated from the model it was
calibrated on is a production incident waiting to happen. Weights, novelty gate,
decision threshold, novelty threshold, family list, and the gate's measured AUC
all travel together in one file.

The measured gate AUC is carried deliberately. The service reads it and, when it
is near chance, labels the novelty field as unproven in its own responses rather
than presenting a number that means nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nxsentinel.data import splits as S  # noqa: E402
from nxsentinel.model.charcnn import CharCNN, CharCNNConfig  # noqa: E402
from nxsentinel.model.ood import MahalanobisOOD, collect_embeddings  # noqa: E402
from nxsentinel.train.evaluate import threshold_at_fpr  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"
PROCESSED = REPO_ROOT / "data" / "processed"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "nogit"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--target-fpr",
        type=float,
        default=0.001,
        help="operating point, calibrated on validation only",
    )
    ap.add_argument(
        "--novelty-percentile",
        type=float,
        default=95.0,
        help="novelty threshold as a percentile of known-family distances",
    )
    args = ap.parse_args()

    ckpt = ARTIFACTS / f"fold{args.fold}.pt"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}, run scripts/train.py first")

    df = pd.read_parquet(PROCESSED / "domains.parquet")
    for c in ("family", "suffix", "origin"):
        df[c] = df[c].astype("category")
    (split,) = S.family_holdout_folds(df, k=args.k, seed=args.seed, only=[args.fold])

    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    model = CharCNN(CharCNNConfig.from_dict(ck["config"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ood = MahalanobisOOD.from_dict(ck["ood"])

    # Both thresholds come from validation. Neither ever sees a test partition.
    val_scores = model.predict_proba(split.val["core"].tolist()).numpy()
    threshold = threshold_at_fpr(split.val["label"], val_scores, args.target_fpr)

    val_dga = split.val.query("label == 1")["core"].tolist()
    known_dist = ood.score(collect_embeddings(model, val_dga))
    novelty_threshold = float(
        pd.Series(known_dist).quantile(args.novelty_percentile / 100.0)
    )

    gate_auc = None
    results = ARTIFACTS / "fold_results.json"
    if results.exists():
        gate_auc = (
            json.loads(results.read_text())
            .get(f"fold{args.fold}", {})
            .get("ood", {})
            .get("auc")
        )

    out = ARTIFACTS / "serving_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.cfg.to_dict(),
            "ood": ood.to_dict(),
            "threshold": float(threshold),
            "novelty_threshold": novelty_threshold,
            "families": list(ck["families"]),
            "gate_auc": gate_auc,
            "version": f"{git_sha()}-fold{args.fold}",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "target_fpr": args.target_fpr,
        },
        out,
    )

    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB, sha256:{digest})")
    print(f"  decision threshold   {threshold:.6f}  (calibrated at {args.target_fpr:.3%} FPR)")
    print(f"  novelty threshold    {novelty_threshold:.4f}  (p{args.novelty_percentile:.0f} of known)")
    print(f"  families             {len(ck['families'])}")
    print(
        f"  gate auc             "
        + (f"{gate_auc:.4f}" if gate_auc else "unknown, run scripts/evaluate.py")
    )
    if gate_auc is not None and gate_auc < 0.70:
        print("  NOTE: gate is below the trust threshold, the API will label it unproven")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
