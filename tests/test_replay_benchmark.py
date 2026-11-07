import pytest

from nxsentinel.ops.benchmark import compare


def _window(device, start, distinct, dga, novelty, combined):
    return {
        "case_id": f"{device}-{start}",
        "device_id": device,
        "window_start": start,
        "window_end": start + 300,
        "distinct_nxdomain": distinct,
        "dga_names": dga,
        "novelty_names": novelty,
        "combined_names": combined,
        "model_version": "test",
    }


def test_replay_benchmark_freezes_thresholds_and_tracks_unreviewed_alerts():
    calibration = [
        _window("a", 0, 9, 3, 3, 4),
        _window("b", 0, 5, 2, 2, 2),
        _window("c", 0, 5, 2, 2, 2),
    ]
    test = [
        _window("a", 86400, 5, 0, 4, 4),
        _window("a", 86700, 5, 0, 4, 4),
        _window("b", 86400, 5, 3, 0, 3),
        _window("c", 86400, 6, 0, 4, 4),
    ]
    infection = {
        "status": "malicious", "episode_id": "episode", "episode_start": 86410,
        "family": "synthetic-dictionary-family", "style": "dictionary",
    }
    labels = [
        {**infection, "case_id": "a-86400"},
        {**infection, "case_id": "a-86700"},
        {"status": "benign", "case_id": "b-86400"},
    ]

    report = compare(calibration, test, labels, 300, 86400, 500)
    assert report["test_device_days"] == 3
    assert report["calibration_alert_slots"] == 1
    cnn = report["methods"]["cnn"]
    fused = report["methods"]["cnn_novelty_behavior"]
    assert cnn["episode_recall"] == 0
    assert cnn["confirmed_false_alerts_per_1000_device_days"] == pytest.approx(1000 / 3)
    assert fused["threshold"] == 3
    assert fused["calibration_alerts"] == 1
    assert fused["alerts"] == 4
    assert fused["unreviewed_alerts"] == 1
    assert fused["reviewed_precision"] == pytest.approx(2 / 3)
    assert fused["episodes_by_style"]["dictionary"] == {"total": 1, "detected": 1}
    assert fused["median_detection_seconds"] == 290


def test_benchmark_rejects_a_mixed_model_comparison():
    a = _window("a", 0, 1, 0, 1, 1)
    b = {**_window("a", 86400, 1, 0, 1, 1), "model_version": "other"}
    with pytest.raises(ValueError, match="same model version"):
        compare([a], [b], [], 300, 86400, 500)
