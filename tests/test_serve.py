"""Service tests.

These run against a synthetic model artifact rather than a trained one, so CI
never needs the corpora or a checkpoint. What is under test is the contract:
routes, validation, batch limits, and the honesty rule that an unproven novelty
gate must say so in the response.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nxsentinel.model.charcnn import CharCNN, CharCNNConfig  # noqa: E402
from nxsentinel.model.ood import MahalanobisOOD  # noqa: E402


def _artifact(tmp_path: Path, gate_auc: float) -> Path:
    model = CharCNN(CharCNNConfig(n_families=3))
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(300, model.cfg.hidden_dim))
    ood = MahalanobisOOD.fit(emb, ["a"] * 100 + ["b"] * 100 + ["c"] * 100)

    p = tmp_path / "serving_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.cfg.to_dict(),
            "ood": ood.to_dict(),
            "threshold": 0.5,
            "novelty_threshold": 100.0,
            "families": ["a", "b", "c"],
            "gate_auc": gate_auc,
            "version": "test",
        },
        p,
    )
    return p


@pytest.fixture
def client(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    monkeypatch.setenv("NXS_MODEL_PATH", str(_artifact(tmp_path, gate_auc=0.53)))
    for m in [k for k in sys.modules if k.startswith("nxsentinel.serve")]:
        del sys.modules[m]
    from nxsentinel.serve.app import app

    with TestClient(app) as c:
        yield c


class TestRoutes:
    def test_healthz_does_not_need_the_model(self, client):
        assert client.get("/healthz").json()["status"] == "ok"

    def test_readyz_reports_the_loaded_model(self, client):
        body = client.get("/readyz").json()
        assert body["status"] == "ready"
        assert body["families"] == 3

    def test_index_serves_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "nxdomain-sentinel" in r.text


class TestScoring:
    def test_single_domain_returns_a_verdict(self, client):
        b = client.post("/v1/score", json={"domain": "mortiscontrastatim.com"}).json()
        assert b["verdict"] in {"dga", "benign"}
        assert 0.0 <= b["dga_probability"] <= 1.0

    def test_public_suffix_is_stripped_from_what_the_model_reads(self, client):
        # The core is the project's central methodological claim, so the API
        # exposes it and this test pins it.
        b = client.post("/v1/score", json={"domain": "cvyh1po636avyrsxebwbkn7.ddns.net"}).json()
        assert b["core"] == "cvyh1po636avyrsxebwbkn7"
        assert "ddns" not in b["core"]

    def test_unparseable_domain_is_reported_not_crashed(self, client):
        b = client.post("/v1/score", json={"domain": "!!!"}).json()
        assert b["verdict"] == "unparseable"
        assert b["note"]

    def test_batch_preserves_every_input(self, client):
        doms = ["github.com", "mortiscontrastatim.com", "!!!", "nan.com"]
        b = client.post("/v1/score/batch", json={"domains": doms}).json()
        assert b["count"] == len(doms)
        assert [r["domain"] for r in b["results"]] == doms

    def test_batch_reports_its_own_latency(self, client):
        b = client.post("/v1/score/batch", json={"domains": ["github.com"]}).json()
        assert b["inference_ms"] > 0

    def test_empty_batch_rejected(self, client):
        assert client.post("/v1/score/batch", json={"domains": []}).status_code == 400

    def test_oversized_batch_rejected(self, client):
        r = client.post("/v1/score/batch", json={"domains": ["a.com"] * 5000})
        assert r.status_code in (413, 422)


class TestHonestyAboutTheGate:
    def test_weak_gate_is_labelled_unproven(self, client):
        # gate_auc is 0.53 in the fixture, near chance. The response must say so
        # rather than presenting is_novel as if it were actionable.
        b = client.post("/v1/score", json={"domain": "mortiscontrastatim.com"}).json()
        if b["verdict"] == "dga":
            assert b["note"] and "experimental" in b["note"]

    def test_strong_gate_carries_no_warning(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        monkeypatch.setenv("NXS_MODEL_PATH", str(_artifact(tmp_path, gate_auc=0.91)))
        for m in [k for k in sys.modules if k.startswith("nxsentinel.serve")]:
            del sys.modules[m]
        from nxsentinel.serve.app import app

        with TestClient(app) as c:
            b = c.post("/v1/score", json={"domain": "mortiscontrastatim.com"}).json()
            assert b["note"] is None
