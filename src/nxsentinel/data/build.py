"""Assemble the three raw corpora into one canonical table.

Output schema, one row per domain:

    core     str   normalised model input, public suffix removed
    suffix   str   the stripped public suffix, audit only, never a feature
    label    int   0 benign, 1 dga
    family   str   DGA family name, or "benign"
    origin   str   which corpus the row came from

Deduplication is global and runs on ``core``. A domain that survives to two
different corpora is kept once, with the training corpus winning, because a
duplicate that lands on both sides of a split silently inflates every number
downstream.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .normalize import normalize_many, read_domain_lines
from .sources import CROSS_BENIGN, CROSS_DGA, TRAIN_CORPUS

BENIGN = "benign"

# nymaim2 is a distinct algorithm but shares a lineage and a name stem with the
# nymaim family in the training corpus. Counting it as "unseen" would overstate
# the cross-source result, so it is excluded from the unseen set by name.
_LINEAGE_COLLISIONS = {"nymaim2": "nymaim"}

# A family needs enough examples for a per-family AUC to mean anything. Below
# this the estimate is too noisy to report, so the family is dropped rather
# than quoted with a misleading number.
MIN_CROSS_FAMILY_N = 50


def load_train_corpus(path: Path) -> pd.DataFrame:
    """Parse the primary corpus, a headerless CSV of label,family,domain."""
    csv = path / "dga_domains_full.csv"
    df = pd.read_csv(csv, header=None, names=["raw_label", "raw_family", "domain"])

    norm, rejects = normalize_many(df["domain"].tolist())
    df["core"] = [n.core for n in norm]
    df["suffix"] = [n.suffix for n in norm]
    df = df[[n.ok for n in norm]].copy()

    df["label"] = (df["raw_label"] == "dga").astype(int)
    df["family"] = df["raw_family"].where(df["label"] == 1, BENIGN)
    df["origin"] = "train_corpus"

    df.attrs["rejects"] = rejects
    return df[["core", "suffix", "label", "family", "origin"]]


def load_cross_dga(path: Path) -> pd.DataFrame:
    """Parse per-family example domains from the generator repository.

    Reads only the checked-in ``example_domains.txt`` files. The generator
    scripts sitting next to them are never imported or run.
    """
    rows: list[dict] = []
    for family_dir in sorted(p for p in path.iterdir() if p.is_dir()):
        if family_dir.name.startswith("."):
            continue
        example = family_dir / "example_domains.txt"
        if not example.exists():
            continue

        text = example.read_text(errors="ignore")
        norm, _ = normalize_many(read_domain_lines(text))
        for n in norm:
            if n.ok:
                rows.append(
                    {
                        "core": n.core,
                        "suffix": n.suffix,
                        "label": 1,
                        "family": family_dir.name,
                        "origin": "cross_dga",
                    }
                )
    return pd.DataFrame(rows, columns=["core", "suffix", "label", "family", "origin"])


def load_cross_benign(path: Path, limit: int = 200_000) -> pd.DataFrame:
    """Parse the independent benign ranking, taking the top ``limit`` entries."""
    listing = path / "top-1000000-domains"
    text = listing.read_text(errors="ignore")

    candidates = []
    for d in read_domain_lines(text):
        candidates.append(d)
        if len(candidates) >= limit:
            break

    norm, _ = normalize_many(candidates)
    rows = [
        {
            "core": n.core,
            "suffix": n.suffix,
            "label": 0,
            "family": BENIGN,
            "origin": "cross_benign",
        }
        for n in norm
        if n.ok
    ]
    return pd.DataFrame(rows, columns=["core", "suffix", "label", "family", "origin"])


def unseen_families(train_df: pd.DataFrame, cross_df: pd.DataFrame) -> list[str]:
    """Families present in the cross corpus and absent from training.

    Name collisions are resolved case-insensitively, and known lineage
    collisions are excluded so a renamed variant of a trained family is not
    counted as a genuinely novel one.
    """
    trained = {f.lower() for f in train_df.loc[train_df.label == 1, "family"].unique()}

    counts = cross_df["family"].value_counts()
    out = []
    for fam, n in counts.items():
        low = fam.lower()
        if low in trained:
            continue
        if _LINEAGE_COLLISIONS.get(low) in trained:
            continue
        if n < MIN_CROSS_FAMILY_N:
            continue
        out.append(fam)
    return sorted(out)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop duplicate cores, preferring the training corpus.

    Also drops any core that appears with both labels, which is a genuine
    annotation conflict rather than a duplicate and cannot be resolved by
    picking a side.
    """
    before = len(df)

    conflicting = df.groupby("core")["label"].nunique()
    conflicted = set(conflicting[conflicting > 1].index)
    df = df[~df["core"].isin(conflicted)].copy()

    priority = {"train_corpus": 0, "cross_dga": 1, "cross_benign": 2}
    df["_prio"] = df["origin"].map(priority).fillna(9)
    df = df.sort_values("_prio", kind="stable").drop_duplicates("core", keep="first")
    df = df.drop(columns="_prio").reset_index(drop=True)

    stats = {
        "rows_in": before,
        "rows_out": len(df),
        "label_conflicts_dropped": len(conflicted),
        "duplicates_dropped": before - len(df) - len(conflicted),
    }
    return df, stats


def build(raw_paths: dict[str, Path], benign_limit: int = 200_000):
    """Build the canonical table from fetched raw sources."""
    train = load_train_corpus(raw_paths[TRAIN_CORPUS.key])
    cross_dga = load_cross_dga(raw_paths[CROSS_DGA.key])
    cross_benign = load_cross_benign(raw_paths[CROSS_BENIGN.key], limit=benign_limit)

    unseen = unseen_families(train, cross_dga)
    cross_dga = cross_dga[cross_dga["family"].isin(unseen)].copy()

    combined = pd.concat([train, cross_dga, cross_benign], ignore_index=True)
    combined, dedup_stats = deduplicate(combined)

    report = {
        "unseen_families": unseen,
        "n_unseen_families": len(unseen),
        "dedup": dedup_stats,
        "train_corpus_rejects": train.attrs.get("rejects", {}),
    }
    return combined, report
