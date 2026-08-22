#!/usr/bin/env python3
"""Paired per-family analysis across all folds.

    python scripts/analyze.py

Fold 0 showed unseen families scoring *higher* than seen ones, which is a
property of which families landed in that holdout rather than a result. Five
families that all generate high-entropy random strings are easy whether or not
you trained on them, while the seen partition kept every hard dictionary-based
family.

This script removes the confound by pairing. Every family is held out in exactly
one fold and trained in the other four, so for each family we can compare its
recall when unseen against its own recall when seen. Family difficulty appears
on both sides of that subtraction and cancels.

It also tests the standing hypothesis about the novelty gate, that the embedding
clusters by generation style rather than family identity, which predicts the
gate should only separate families whose style is absent from training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

ARTIFACTS = REPO_ROOT / "artifacts"

# Style assignment is from the published characterisation of each algorithm,
# fixed before any result was inspected so it cannot be fitted to the outcome.
STYLE = {
    "suppobox": "dictionary", "matsnu": "dictionary", "gozi": "dictionary",
    "banjori": "dictionary", "pizd": "dictionary",
    "cryptolocker": "random", "dircrypt": "random", "padcrypt": "random",
    "ramnit": "random", "rovnix": "random", "emotet": "random",
    "pykspa": "random", "ranbyus": "random", "tinba": "random",
    "conficker": "random", "corebot": "random", "fobber": "random",
    "kraken": "random", "murofet": "random", "necurs": "random",
    "nymaim": "random", "qadars": "random", "ramdo": "random",
    "simda": "random", "symmi": "random", "vawtrak": "random",
    "pushdo": "random",
}


def load_results() -> dict:
    p = ARTIFACTS / "fold_results.json"
    if not p.exists():
        raise SystemExit("no results, run scripts/evaluate.py first")
    return json.loads(p.read_text())


def paired_family_table(res: dict) -> list[dict]:
    """For each family, recall when held out versus recall when trained.

    ``test_unseen`` per-family entries come from the single fold that held the
    family out. Seen-side recall is read from the other folds, where the same
    family sat in training and its rows landed in that fold's in-distribution
    reference partition.
    """
    unseen: dict[str, float] = {}
    seen: dict[str, list[float]] = {}

    for _, r in sorted(res.items()):
        held = set(r["meta"]["holdout_families"])
        for fam, m in r["test_unseen"].get("per_family", {}).items():
            if fam in held:
                unseen[fam] = m["recall"]
        for fam, m in r.get("test_seen_per_family", {}).items():
            if fam not in held:
                seen.setdefault(fam, []).append(m["recall"])

    rows = []
    for fam in sorted(set(unseen) | set(seen)):
        s = seen.get(fam, [])
        u = unseen.get(fam)
        if u is None or not s:
            continue
        mean_seen = float(np.mean(s))
        rows.append(
            {
                "family": fam,
                "style": STYLE.get(fam, "unknown"),
                "recall_seen": mean_seen,
                "recall_unseen": u,
                "cost_of_holdout": mean_seen - u,
                "n_seen_folds": len(s),
            }
        )
    return rows


def gate_by_style(res: dict) -> list[dict]:
    """Novelty gate AUC per fold, annotated with the styles it held out.

    The hypothesis predicts a fold holding out a dictionary family should score
    markedly higher than folds holding out only random-string families.
    """
    rows = []
    for name, r in sorted(res.items()):
        held = r["meta"]["holdout_families"]
        styles = sorted({STYLE.get(f, "unknown") for f in held})
        rows.append(
            {
                "fold": name,
                "gate_auc": r["ood"]["auc"],
                "held_styles": styles,
                "has_dictionary": "dictionary" in styles,
                "holdout": held,
            }
        )
    return rows


def main() -> int:
    res = load_results()
    if len(res) < 2:
        print(f"only {len(res)} fold(s) evaluated, the paired analysis needs all folds")

    rows = paired_family_table(res)

    print("\nPAIRED PER-FAMILY RECALL AT 0.1% FPR")
    print("Same family, held out once and trained four times. Difficulty cancels.\n")
    if not rows:
        print("  no families with both a seen and an unseen measurement yet")
    else:
        print(f"  {'family':<15}{'style':<12}{'seen':>9}{'unseen':>9}{'cost':>9}")
        print(f"  {'-' * 54}")
        for r in sorted(rows, key=lambda x: -x["cost_of_holdout"]):
            print(
                f"  {r['family']:<15}{r['style']:<12}{r['recall_seen']:>9.4f}"
                f"{r['recall_unseen']:>9.4f}{r['cost_of_holdout']:>+9.4f}"
            )
        costs = np.array([r["cost_of_holdout"] for r in rows])
        print(f"\n  mean cost of holding a family out: {costs.mean():+.4f}")
        print(f"  median: {np.median(costs):+.4f}   worst: {costs.max():+.4f}")

    print("\n\nNOVELTY GATE BY HELD-OUT STYLE")
    print("Hypothesis: the gate only fires when the held-out *style* is novel.\n")
    gates = gate_by_style(res)
    print(f"  {'fold':<8}{'gate AUC':>10}  {'dictionary held out?':<22}holdout")
    print(f"  {'-' * 76}")
    for g in gates:
        print(
            f"  {g['fold']:<8}{g['gate_auc']:>10.4f}  "
            f"{'yes' if g['has_dictionary'] else 'no':<22}{', '.join(g['holdout'])}"
        )

    dict_folds = [g["gate_auc"] for g in gates if g["has_dictionary"]]
    rand_folds = [g["gate_auc"] for g in gates if not g["has_dictionary"]]
    if dict_folds and rand_folds:
        d, r = float(np.mean(dict_folds)), float(np.mean(rand_folds))
        print(f"\n  mean AUC, folds holding out a dictionary family: {d:.4f}")
        print(f"  mean AUC, folds holding out only random families: {r:.4f}")
        print(f"  difference: {d - r:+.4f}")
        print(
            "\n  VERDICT: "
            + (
                "consistent with the style hypothesis"
                if d - r > 0.10
                else "does NOT support the style hypothesis, the gate needs redesign"
            )
        )
    else:
        print("\n  need folds of both kinds to test this")

    out = ARTIFACTS / "analysis.json"
    out.write_text(
        json.dumps({"paired_families": rows, "gate_by_style": gates}, indent=2, default=float)
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
