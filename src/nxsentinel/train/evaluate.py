"""Metrics.

Two choices here are deliberate and worth stating.

Thresholds are picked on validation and then frozen. A per-test-set threshold
would quietly tune on the very families the experiment claims are unseen.

The headline operating point is recall at a fixed low false positive rate rather
than accuracy or F1. Domain traffic is overwhelmingly benign, so a detector at
1% FPR generates an unusable flood of alerts. Reporting recall at 0.1% FPR is
the number that reflects whether anyone could run this.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(y_true, scores, target_fpr: float) -> float:
    """Smallest threshold whose false positive rate is at or below target."""
    neg = np.asarray(scores)[np.asarray(y_true) == 0]
    if len(neg) == 0:
        return 0.5
    return float(np.quantile(neg, 1.0 - target_fpr))


def binary_metrics(y_true, scores, thresholds: dict[str, float]) -> dict:
    y = np.asarray(y_true)
    s = np.asarray(scores)

    out = {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    }
    if out["n_pos"] and out["n_neg"]:
        out["auc"] = float(roc_auc_score(y, s))
        out["auprc"] = float(average_precision_score(y, s))
    else:
        out["auc"] = out["auprc"] = float("nan")

    for name, thr in thresholds.items():
        pred = s >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        out[f"recall@{name}"] = tp / max(tp + fn, 1)
        out[f"fpr@{name}"] = fp / max(out["n_neg"], 1)
    return out


def per_family_recall(df, scores, threshold: float) -> dict[str, dict]:
    """Recall broken out by family, so one easy family cannot carry the mean."""
    s = np.asarray(scores)
    dga = df["label"].to_numpy() == 1
    out = {}
    for fam in sorted(df.loc[dga, "family"].unique()):
        mask = dga & (df["family"].to_numpy() == fam)
        n = int(mask.sum())
        if not n:
            continue
        out[fam] = {"n": n, "recall": float((s[mask] >= threshold).mean())}
    return out


def ood_separation_per_family(seen_scores, unseen_df, unseen_scores) -> dict:
    """Gate separation computed one held-out family at a time.

    Pooling every held-out family into a single AUC destroys the signal this
    gate exists to produce. A fold holding out four stylistically familiar
    families and one genuinely novel one averages to chance, and the pooled
    number then reads as "the gate does not work" when what it actually means
    is "four of these five families are not novel, and the gate said so".

    Reporting per family is not a presentational choice. It is the only
    resolution at which the measurement answers the question.
    """
    seen = np.asarray(seen_scores)
    out = {}
    if len(seen) == 0:
        return out

    thr = float(np.quantile(seen, 0.95))
    fams = unseen_df["family"].astype(str).to_numpy()
    scores = np.asarray(unseen_scores)

    for fam in sorted(set(fams.tolist())):
        d = scores[fams == fam]
        if len(d) < 50:
            continue
        y = np.concatenate([np.zeros(len(seen)), np.ones(len(d))])
        out[fam] = {
            "n": int(len(d)),
            "auc": float(roc_auc_score(y, np.concatenate([seen, d]))),
            "flagged_at_5pct_seen_fpr": float((d >= thr).mean()),
            "median_distance": float(np.median(d)),
        }
    return out


def ood_separation(seen_scores, unseen_scores) -> dict:
    """How well the gate separates unseen families from trained ones.

    AUC of 0.5 means the gate is blind and the embedding carries no novelty
    signal. Above 0.5 means distance from known centroids is informative about
    whether a family was in training.
    """
    seen = np.asarray(seen_scores)
    unseen = np.asarray(unseen_scores)
    if len(seen) == 0 or len(unseen) == 0:
        return {"auc": float("nan")}

    y = np.concatenate([np.zeros(len(seen)), np.ones(len(unseen))])
    s = np.concatenate([seen, unseen])
    thr = float(np.quantile(seen, 0.95))
    return {
        "auc": float(roc_auc_score(y, s)),
        "unseen_flagged_at_5pct_seen_fpr": float((unseen >= thr).mean()),
        "median_seen_distance": float(np.median(seen)),
        "median_unseen_distance": float(np.median(unseen)),
    }


def evaluate_fold(model, ood, split, cross) -> dict:
    """Score a trained fold on its seen, unseen, and cross-source partitions.

    Thresholds come from validation and are then frozen across all three, so
    the unseen-family number is never tuned on unseen families.
    """
    import pandas as pd

    from ..model.ood import collect_embeddings

    val_scores = model.predict_proba(split.val["core"].tolist()).numpy()
    thresholds = {
        "fpr0.1%": threshold_at_fpr(split.val["label"], val_scores, 0.001),
        "fpr1%": threshold_at_fpr(split.val["label"], val_scores, 0.01),
    }
    res = {"thresholds": {k: float(v) for k, v in thresholds.items()}}

    parts = {
        "test_seen": split.extra["test_seen"],
        "test_unseen": split.test,
        "cross_source": pd.concat(
            [cross["cross_dga"], cross["cross_benign"]], ignore_index=True
        ),
    }
    known_validation = split.val.loc[split.val["label"] == 1, "core"].tolist()
    novelty_threshold = float(np.quantile(
        ood.score(collect_embeddings(model, known_validation)), 0.95
    ))
    res["novelty_threshold"] = novelty_threshold
    res["benign_novelty"] = {}
    for name, df in parts.items():
        scores = model.predict_proba(df["core"].tolist()).numpy()
        res[name] = binary_metrics(df["label"], scores, thresholds)
        benign = df["label"].to_numpy() == 0
        distances = ood.score(collect_embeddings(model, df.loc[benign, "core"].tolist()))
        high_novelty = distances >= novelty_threshold
        res["benign_novelty"][name] = {
            "n_benign": int(benign.sum()),
            "above_novelty_threshold": int(high_novelty.sum()),
            "above_both_thresholds": int(
                (high_novelty & (scores[benign] >= thresholds["fpr0.1%"])).sum()
            ),
        }
        fam = per_family_recall(df, scores, thresholds["fpr0.1%"])
        if name == "test_seen":
            # Hoisted to the top level because the paired analysis reads it as
            # the seen-side measurement for families this fold trained on.
            res["test_seen_per_family"] = fam
        else:
            res[name]["per_family"] = fam

    seen_df = split.extra["test_seen"].query("label == 1")
    unseen_df = split.test.query("label == 1")
    seen = ood.score(collect_embeddings(model, seen_df["core"].tolist()))
    unseen = ood.score(collect_embeddings(model, unseen_df["core"].tolist()))

    res["ood"] = ood_separation(seen, unseen)
    res["ood_per_family"] = ood_separation_per_family(seen, unseen_df, unseen)
    return res
