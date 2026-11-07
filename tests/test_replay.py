import json
from collections import Counter
from types import SimpleNamespace

import pytest

from nxsentinel.ops.replay import alert, device_id, events, windows


def _line(ts, client, query, rcode=3):
    return json.dumps({"ts": ts, "id.orig_h": client, "query": query, "rcode": rcode})


def test_replay_groups_devices_keeps_counts_and_scores_unique_names():
    lines = [
        _line(10, "a", "Novel.COM."),
        _line(11, "a", "novel.com"),
        _line(12, "b", "novel.com"),
        _line(13, "a", "dga.com"),
        _line(14, "a", "success.com", 0),
        _line(15, "a", "servfail.com", 2),
        _line(300, "a", "novel.com"),
    ]
    batches = []

    def predict(names):
        batches.append(names)
        return [
            SimpleNamespace(verdict="dga" if name == "dga.com" else "benign",
                            above_novelty_threshold=name == "novel.com",
                            dga_probability=0.5, novelty_score=1.0)
            for name in names
        ]

    counts = Counter()
    secret = b"k" * 32
    result = list(windows(events(lines, secret, counts), predict, batch_size=1))
    by_key = {(row["device_id"], row["window_start"]): row for row in result}
    row = by_key[device_id(secret, "a"), 0]
    assert row["responses"] == 5
    assert row["nxdomain_responses"] == 3
    assert row["distinct_nxdomain"] == row["combined_names"] == 2
    assert row["dga_names"] == row["novelty_names"] == 1
    assert row["evidence"][1]["count"] == 2
    assert alert(row, min_distinct=2, min_suspicious=2)
    assert not alert(by_key[device_id(secret, "b"), 0], min_distinct=2, min_suspicious=2)
    assert batches == [["dga.com"], ["novel.com"], ["novel.com"]]
    assert counts == {"responses": 7, "nxdomain_responses": 5}


def test_replay_reports_out_of_order_line():
    lines = [_line(2, "a", "a.com"), _line(1, "a", "b.com")]
    with pytest.raises(ValueError, match="line 2: timestamps"):
        list(events(lines, b"k" * 32, Counter()))
