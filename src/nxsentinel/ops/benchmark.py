"""Forward-in-time comparisons at a shared calibration alert budget."""

from collections import Counter, defaultdict
from math import floor, isfinite
from statistics import median

METHODS = {
    "behavior": "distinct_nxdomain",
    "cnn": "dga_names",
    "novelty": "novelty_names",
    "cnn_novelty": "combined_names",
    "cnn_novelty_behavior": "combined_names",
}


def _device_days(rows):
    return len({(row["device_id"], row["window_start"] // 86400) for row in rows})


def _scores(rows, method, min_distinct):
    return [
        row[METHODS[method]]
        if method != "cnn_novelty_behavior" or row["distinct_nxdomain"] >= min_distinct
        else 0
        for row in rows
    ]


def _threshold(scores, budget):
    """Include whole score ties only when they fit the calibration budget."""
    counts = Counter(score for score in scores if score > 0)
    if not counts:
        return 1
    for score in sorted(counts, reverse=True):
        if counts[score] > budget:
            return score + 1
        budget -= counts[score]
    return 1


def _label_index(labels, test_ids):
    by_case = {}
    episodes = {}
    for label in labels:
        case = label["case_id"]
        if case in by_case:
            raise ValueError("duplicate case ID in labels")
        if label["status"] not in {"benign", "malicious"}:
            raise ValueError("status must be benign or malicious")
        by_case[case] = label
        if label["status"] == "malicious":
            episode = label["episode_id"]
            metadata = (label["family"], label["style"], label["episode_start"])
            if not isfinite(metadata[2]) or metadata[2] < 0:
                raise ValueError("episode_start must be finite nonnegative Unix seconds")
            if episodes.setdefault(episode, metadata) != metadata:
                raise ValueError("inconsistent family, style or start for an episode")
    if not by_case.keys() & test_ids:
        raise ValueError("no supplied case labels match the test windows")
    return by_case, episodes


def _metrics(alerts, by_case, eligible, episodes, days):
    reviewed = [row for row in alerts if row["case_id"] in by_case]
    malicious = [row for row in reviewed if by_case[row["case_id"]]["status"] == "malicious"]
    first_detection = {}
    for row in malicious:
        label = by_case[row["case_id"]]
        episode = label["episode_id"]
        if episode in eligible:
            first_detection[episode] = min(
                row["window_end"], first_detection.get(episode, row["window_end"])
            )

    family = defaultdict(Counter)
    style = defaultdict(Counter)
    for episode in eligible:
        family_name, style_name, _ = episodes[episode]
        for breakdown, name in ((family, family_name), (style, style_name)):
            breakdown[name]["total"] += 1
            breakdown[name]["detected"] += episode in first_detection

    false_alerts = len(reviewed) - len(malicious)
    return {
        "alerts": len(alerts),
        "alerts_per_1000_device_days": 1000 * len(alerts) / days,
        "unreviewed_alerts": len(alerts) - len(reviewed),
        "reviewed_precision": len(malicious) / len(reviewed) if reviewed else None,
        "confirmed_false_alerts_per_1000_device_days": 1000 * false_alerts / days,
        "episode_recall": len(first_detection) / len(eligible) if eligible else None,
        "median_detection_seconds": median(
            detected - episodes[episode][2] for episode, detected in first_detection.items()
        ) if first_detection else None,
        "episodes_by_family": dict(sorted(family.items())),
        "episodes_by_style": dict(sorted(style.items())),
    }


def compare(
    calibration_rows, test_rows, labels, calibration_end, test_start,
    alert_budget, min_distinct=5,
):
    """Freeze thresholds on earlier windows, then score later labeled windows."""
    if not 0 < calibration_end <= test_start or not isfinite(test_start):
        raise ValueError("require positive calibration end <= test start")
    if not isfinite(alert_budget) or alert_budget < 0 or min_distinct < 1:
        raise ValueError("require a finite nonnegative alert budget and positive burst threshold")
    calibration = [row for row in calibration_rows if row["window_end"] <= calibration_end]
    test = [row for row in test_rows if row["window_start"] >= test_start]
    if not calibration or not test:
        raise ValueError("both calibration and test need at least one eligible window")
    keys = [row["case_id"] for row in calibration + test]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate case ID in feature windows")
    model_versions = {row["model_version"] for row in calibration + test}
    if len(model_versions) != 1:
        raise ValueError("calibration and test must use the same model version")
    window_sizes = {row["window_end"] - row["window_start"] for row in calibration + test}
    if len(window_sizes) != 1 or min(window_sizes) <= 0:
        raise ValueError("calibration and test must use the same positive window size")
    calibration_days, test_days = _device_days(calibration), _device_days(test)
    slots = floor(alert_budget * calibration_days / 1000)
    by_case, episodes = _label_index(labels, {row["case_id"] for row in test})
    eligible = set()
    for row in test:
        label = by_case.get(row["case_id"])
        if label and label["status"] == "malicious":
            if label["episode_start"] >= row["window_end"]:
                raise ValueError("malicious case window ends before its episode starts")
            if label["episode_start"] >= test_start:
                eligible.add(label["episode_id"])

    report = {
        "calibration_device_days": calibration_days,
        "test_device_days": test_days,
        "labeled_test_windows": sum(row["case_id"] in by_case for row in test),
        "eligible_test_episodes": len(eligible),
        "calibration_alert_budget_per_1000_device_days": alert_budget,
        "calibration_alert_slots": slots,
        "model_version": next(iter(model_versions)),
        "methods": {},
    }
    for method in METHODS:
        scores = _scores(calibration, method, min_distinct)
        threshold = _threshold(scores, slots)
        alerts = [
            row for row, score in zip(test, _scores(test, method, min_distinct), strict=True)
            if score >= threshold
        ]
        report["methods"][method] = {
            "threshold": threshold,
            "calibration_alerts": sum(score >= threshold for score in scores),
            **_metrics(alerts, by_case, eligible, episodes, test_days),
        }
    return report
