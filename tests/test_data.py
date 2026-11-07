"""Tests for normalisation and split integrity.

The leakage tests matter more than they look. A split bug does not crash, it
just quietly raises every score, so these assert the failure is detected rather
than trusting that it never happens.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nxsentinel.data.normalize import normalize, read_domain_lines  # noqa: E402
from nxsentinel.data.splits import (  # noqa: E402
    LeakageError,
    Split,
    assert_no_leakage,
    family_folds,
    family_holdout_folds,
    random_split,
)


class TestNormalize:
    def test_strips_plain_tld(self):
        assert normalize("mortiscontrastatim.com").core == "mortiscontrastatim"

    def test_strips_private_suffix_so_ddns_is_not_a_feature(self):
        # The whole point. corebot hides behind a dynamic DNS provider, and the
        # provider name must not survive into the model input.
        n = normalize("cvyh1po636avyrsxebwbkn7.ddns.net")
        assert n.core == "cvyh1po636avyrsxebwbkn7"
        assert n.suffix == "ddns.net"
        assert "ddns" not in n.core

    def test_strips_multipart_suffix(self):
        assert normalize("plasticbags.sa.com").core == "plasticbags"

    def test_keeps_subdomain_structure(self):
        n = normalize("mail.corp.example.com")
        assert n.core == "mail.corp.example"

    def test_case_and_trailing_dot(self):
        assert normalize("EXAMPLE.COM.").core == "example"

    @pytest.mark.parametrize(
        "bad,reason",
        [
            ("", "empty"),
            ("xn--80ak6aa92e.com", "punycode"),
            ("has space.com", "bad_chars"),
            ("nosuffixatall", "no_public_suffix"),
            ("a.com", "too_short"),
        ],
    )
    def test_rejects(self, bad, reason):
        n = normalize(bad)
        assert not n.ok
        assert n.reason == reason

    def test_rejects_overlong(self):
        assert not normalize("a" * 200 + ".com").ok

    def test_read_lines_handles_rank_prefix(self):
        text = "1,google.com\n2,youtube.com\n# comment\n\nbare.org\n"
        assert list(read_domain_lines(text)) == [
            "google.com",
            "youtube.com",
            "bare.org",
        ]


class TestNaLikeDomains:
    """nan.com and friends are real registrations, not missing values."""

    @pytest.mark.parametrize("core", ["nan", "inf", "na", "null", "none", "-inf"])
    def test_na_like_cores_encode_without_coercion(self, core):
        from nxsentinel.model.charcnn import encode_array

        arr = encode_array([core])
        assert arr.shape[0] == 1
        assert arr[0].sum() > 0

    def test_encoder_survives_a_float_leaking_through(self):
        from nxsentinel.model.charcnn import encode_array

        # Reproduces the CSV fallback failure rather than trusting it is fixed.
        assert encode_array([float("nan")]).shape == (1, 64)

    def test_csv_round_trip_preserves_na_like_cores(self, tmp_path):
        rows = pd.DataFrame(
            {
                "core": ["nan", "inf", "null", "ordinary"],
                "suffix": ["com"] * 4,
                "label": [0, 0, 0, 1],
                "family": ["benign"] * 3 + ["fam0"],
                "origin": ["train_corpus"] * 4,
            }
        )
        f = tmp_path / "d.csv.gz"
        rows.to_csv(f, index=False, compression="gzip")

        back = pd.read_csv(
            f,
            dtype={"core": str, "suffix": str, "family": str, "origin": str},
            keep_default_na=False,
        )
        assert back["core"].tolist() == ["nan", "inf", "null", "ordinary"]


def _toy(n_fam: int = 10, per_fam: int = 40, n_benign: int = 400) -> pd.DataFrame:
    rows = []
    for f in range(n_fam):
        for i in range(per_fam):
            rows.append(
                {
                    "core": f"dga{f}x{i}",
                    "suffix": "com",
                    "label": 1,
                    "family": f"fam{f}",
                    "origin": "train_corpus",
                }
            )
    for i in range(n_benign):
        rows.append(
            {
                "core": f"benign{i}",
                "suffix": "com",
                "label": 0,
                "family": "benign",
                "origin": "train_corpus",
            }
        )
    return pd.DataFrame(rows)


class TestSplits:
    def test_leakage_detector_actually_fires(self):
        shared = pd.DataFrame(
            [{"core": "dup", "suffix": "com", "label": 1, "family": "f", "origin": "train_corpus"}]
        )
        bad = Split(name="bad", train=shared, val=shared.iloc[:0], test=shared)
        with pytest.raises(LeakageError):
            assert_no_leakage(bad)

    def test_random_split_is_clean(self):
        assert_no_leakage(random_split(_toy(), seed=1))

    def test_folds_partition_every_family_exactly_once(self):
        df = _toy(n_fam=10)
        groups = family_folds(df, k=5, seed=3)
        flat = [f for g in groups for f in g]
        assert sorted(flat) == sorted(df.loc[df.label == 1, "family"].unique())
        assert len(flat) == len(set(flat))

    def test_holdout_families_never_appear_in_train_or_val(self):
        df = _toy(n_fam=10)
        for split in family_holdout_folds(df, k=5, seed=3):
            held = set(split.meta["holdout_families"])
            assert not held & set(split.train["family"])
            assert not held & set(split.val["family"])
            assert held <= set(split.test["family"])

    def test_every_fold_is_leak_free(self):
        for split in family_holdout_folds(_toy(n_fam=10), k=5, seed=3):
            assert_no_leakage(split)

    def test_folds_are_deterministic_under_seed(self):
        df = _toy(n_fam=10)
        a = [s.meta["holdout_families"] for s in family_holdout_folds(df, k=5, seed=7)]
        b = [s.meta["holdout_families"] for s in family_holdout_folds(df, k=5, seed=7)]
        c = [s.meta["holdout_families"] for s in family_holdout_folds(df, k=5, seed=8)]
        assert a == b
        assert a != c

    def test_cross_source_rows_never_enter_training(self):
        df = _toy(n_fam=10)
        cross = pd.DataFrame(
            [
                {
                    "core": f"cross{i}",
                    "suffix": "com",
                    "label": 1,
                    "family": "novel",
                    "origin": "cross_dga",
                }
                for i in range(50)
            ]
        )
        combined = pd.concat([df, cross], ignore_index=True)
        for split in family_holdout_folds(combined, k=5, seed=3):
            for part in (split.train, split.val, split.test):
                assert "cross_dga" not in set(part["origin"])

    def test_partitions_cover_the_pool_exactly(self):
        # Every row is used exactly once. The remainder-absorbing cut means
        # integer truncation cannot silently drop rows on the floor.
        df = _toy(n_fam=10)
        for split in family_holdout_folds(df, k=5, seed=3):
            total = sum(len(p) for p in split.partitions().values())
            assert total == len(df)

    def test_test_partition_holds_only_holdout_families(self):
        # The headline number is an unseen-family number, so a single seen
        # family row reaching the test partition would quietly inflate it.
        df = _toy(n_fam=10)
        for split in family_holdout_folds(df, k=5, seed=3):
            held = set(split.meta["holdout_families"])
            dga_fams = set(split.test.loc[split.test.label == 1, "family"])
            assert dga_fams == held

    def test_seen_reference_holds_no_holdout_family(self):
        df = _toy(n_fam=10)
        for split in family_holdout_folds(df, k=5, seed=3):
            held = set(split.meta["holdout_families"])
            seen_fams = set(split.extra["test_seen"]["family"])
            assert not held & seen_fams
