"""Splitting strategy. This is the file the headline result depends on.

Two split modes exist here, and the contrast between them is the point of the
whole project.

random_split
    The conventional shuffle. Every family appears in train and in test. Models
    score above 0.99 on it, which is why nearly every DGA paper reports a number
    like that, and why the number tells a defender almost nothing.

family_holdout_folds
    Whole families are removed from training and only reappear at test time.
    This asks the question a SOC actually has, which is whether the detector
    fires on a family that did not exist when the model was trained.

Both modes share one hard rule. A normalised core string may never appear on
both sides of a split. ``assert_no_leakage`` enforces it and is called on every
split this module produces, because a silent leak here would make every
downstream number wrong in the flattering direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BENIGN = "benign"


@dataclass
class Split:
    """One train/val/test partition, plus enough metadata to reproduce it.

    ``extra`` holds additional evaluation-only partitions. The family-holdout
    mode uses it for ``test_seen``, a held-out slice of the families the model
    did train on. Scoring both with one model gives the in-distribution versus
    unseen-family contrast without the confound of comparing two separately
    trained models.
    """

    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    meta: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def partitions(self) -> dict:
        """Every disjoint partition in this split, keyed by name."""
        return {"train": self.train, "val": self.val, "test": self.test, **self.extra}

    def summary(self) -> dict:
        def part(d: pd.DataFrame) -> dict:
            return {
                "n": int(len(d)),
                "n_dga": int((d.label == 1).sum()),
                "n_benign": int((d.label == 0).sum()),
                "n_families": int(d.loc[d.label == 1, "family"].nunique()),
            }

        out = {"name": self.name}
        out.update({k: part(v) for k, v in self.partitions().items()})
        out.update(self.meta)
        return out


class LeakageError(AssertionError):
    """Raised when the same core string reaches two partitions of one split."""


def assert_no_leakage(split: Split) -> None:
    """Verify no core string appears in two partitions of the same split."""
    parts = {k: set(v["core"]) for k, v in split.partitions().items()}
    names = list(parts)

    for i, a_name in enumerate(names):
        for b_name in names[i + 1 :]:
            overlap = parts[a_name] & parts[b_name]
            if overlap:
                sample = sorted(overlap)[:5]
                raise LeakageError(
                    f"{split.name}: {len(overlap)} cores shared between "
                    f"{a_name} and {b_name}, e.g. {sample}"
                )


def _cut(df: pd.DataFrame, rng: np.random.Generator, fracs) -> list[pd.DataFrame]:
    """Shuffle a frame and cut it into disjoint parts sized by ``fracs``.

    The final part absorbs any remainder from integer truncation, so the parts
    always reconstruct the input exactly and no rows are silently discarded.
    """
    d = df.iloc[rng.permutation(len(df))]
    out, start = [], 0
    for f in fracs[:-1]:
        n = int(len(df) * f)
        out.append(d.iloc[start : start + n].copy())
        start += n
    out.append(d.iloc[start:].copy())
    return out


def _three_way(
    df: pd.DataFrame, rng: np.random.Generator, fracs=(0.60, 0.15, 0.25)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a, b, c = _cut(df, rng, fracs)
    return a, b, c


def _training_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Rows eligible for training. Cross-source rows are evaluation only.

    Keeping cross-source data out of every training pool is what lets the
    cross-source result be quoted as a genuine out-of-sample number.
    """
    return df[df["origin"] == "train_corpus"].copy()


def random_split(df: pd.DataFrame, seed: int = 0) -> Split:
    """Conventional shuffled split. The in-distribution baseline."""
    pool = _training_pool(df)
    rng = np.random.default_rng(seed)
    train, val, test = _three_way(pool, rng)

    split = Split(
        name="random",
        train=train,
        val=val,
        test=test,
        meta={"mode": "random", "seed": seed},
    )
    assert_no_leakage(split)
    return split


def family_folds(df: pd.DataFrame, k: int = 5, seed: int = 0) -> list[list[str]]:
    """Partition DGA families into k holdout groups of near-equal size.

    Families are shuffled before dealing so that fold composition does not
    track alphabetical order or corpus order, either of which would correlate
    with algorithm style and bias the folds.
    """
    pool = _training_pool(df)
    fams = sorted(pool.loc[pool.label == 1, "family"].unique())

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(fams))
    shuffled = [fams[i] for i in order]

    groups: list[list[str]] = [[] for _ in range(k)]
    for i, fam in enumerate(shuffled):
        groups[i % k].append(fam)
    return [sorted(g) for g in groups]


def family_holdout_folds(
    df: pd.DataFrame, k: int = 5, seed: int = 0, only: list[int] | None = None
) -> list[Split]:
    """Build k splits, each holding out a disjoint group of whole families.

    Across the k folds every family is held out exactly once, so each family
    contributes one unseen-family measurement and the aggregate covers the
    entire corpus rather than one lucky subset.

    ``only`` restricts which folds are materialised. Each Split holds copies of
    its partitions, so building all five at once duplicates the corpus roughly
    five times over, which was enough to exhaust a 4 GB instance before
    training started. Fold composition and per-fold seeding are independent of
    this argument, so a fold built alone is byte for byte the fold that would
    have been built alongside the rest.
    """
    wanted = set(range(k)) if only is None else set(only)
    pool = _training_pool(df)
    groups = family_folds(df, k=k, seed=seed)

    dga = pool[pool.label == 1]
    benign = pool[pool.label == 0]

    splits: list[Split] = []
    for fold_i, holdout in enumerate(groups):
        if fold_i not in wanted:
            continue
        rng = np.random.default_rng(seed * 1000 + fold_i)

        seen = dga[~dga["family"].isin(holdout)]
        unseen = dga[dga["family"].isin(holdout)]

        # Seen families are cut three ways. Train and val drive fitting and
        # model selection. The third slice becomes test_seen, an in-distribution
        # reference scored by the same model that is scored on the unseen
        # families, which is what makes the two numbers directly comparable.
        seen_train, seen_val, seen_test = _cut(seen, rng, (0.70, 0.15, 0.15))
        ben_train, ben_val, ben_test, ben_seen = _cut(
            benign, rng, (0.55, 0.15, 0.15, 0.15)
        )

        train = pd.concat([seen_train, ben_train], ignore_index=True)
        val = pd.concat([seen_val, ben_val], ignore_index=True)
        # test holds held-out families only. No seen-family row reaches it, so
        # its score is an unseen-family score with nothing else mixed in.
        test = pd.concat([unseen, ben_test], ignore_index=True)
        test_seen = pd.concat([seen_test, ben_seen], ignore_index=True)

        split = Split(
            name=f"family_holdout_fold{fold_i}",
            train=train,
            val=val,
            test=test,
            extra={"test_seen": test_seen},
            meta={
                "mode": "family_holdout",
                "fold": fold_i,
                "seed": seed,
                "holdout_families": holdout,
                "n_holdout_families": len(holdout),
            },
        )
        assert_no_leakage(split)
        splits.append(split)

    return splits


def cross_source_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Evaluation-only frames from corpora the model never trains on."""
    return {
        "cross_dga": df[df["origin"] == "cross_dga"].copy(),
        "cross_benign": df[df["origin"] == "cross_benign"].copy(),
    }


def assert_cross_source_clean(df: pd.DataFrame) -> None:
    """Verify cross-source rows share no core with the training pool."""
    pool = set(_training_pool(df)["core"])
    for name, part in cross_source_sets(df).items():
        overlap = pool & set(part["core"])
        if overlap:
            raise LeakageError(
                f"{name}: {len(overlap)} cores also present in the training pool"
            )
