"""Inference wrapper.

Holds the three things a verdict needs, the classifier, the novelty gate, and
the thresholds calibrated on validation. Keeping them in one artifact matters,
because a threshold separated from the model it was calibrated on is a
production incident waiting to happen.

The novelty flag is reported honestly. If the gate's measured separation is at
or near chance, ``is_novel`` is still returned but ``note`` says the gate is not
currently reliable. Silently shipping a field that means nothing is worse than
shipping one labelled as unproven.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.normalize import normalize
from ..model.charcnn import CharCNN, CharCNNConfig, encode_batch
from ..model.ood import MahalanobisOOD

# Below this measured AUC the gate is not separating novel families from known
# ones well enough to act on, and the response says so.
GATE_TRUSTWORTHY_AUC = 0.70


@dataclass
class Verdict:
    domain: str
    core: str
    verdict: str
    dga_probability: float
    nearest_family: str | None
    novelty_score: float | None
    is_novel: bool
    above_novelty_threshold: bool | None = None
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "core": self.core,
            "verdict": self.verdict,
            "dga_probability": round(self.dga_probability, 6),
            "nearest_family": self.nearest_family,
            "novelty_score": (
                round(self.novelty_score, 4) if self.novelty_score is not None else None
            ),
            "is_novel": self.is_novel,
            "above_novelty_threshold": self.above_novelty_threshold,
            "note": self.note,
        }


class Predictor:
    def __init__(
        self,
        model: CharCNN,
        ood: MahalanobisOOD,
        threshold: float,
        novelty_threshold: float,
        families: list[str],
        version: str,
        gate_auc: float | None = None,
        artifact_sha256: str | None = None,
    ):
        self.model = model.eval()
        self.ood = ood
        self.threshold = threshold
        self.novelty_threshold = novelty_threshold
        self.families = families
        self.version = version
        self.gate_auc = gate_auc
        self.artifact_sha256 = artifact_sha256

    @property
    def gate_is_trustworthy(self) -> bool:
        return self.gate_auc is not None and self.gate_auc >= GATE_TRUSTWORTHY_AUC

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str | None = None) -> "Predictor":
        artifact = Path(path).read_bytes()
        digest = hashlib.sha256(artifact).hexdigest()
        if expected_sha256 is not None and not hmac.compare_digest(
            digest, expected_sha256.strip().lower()
        ):
            raise ValueError("model artifact SHA-256 mismatch")
        ck = torch.load(io.BytesIO(artifact), weights_only=True, map_location="cpu")
        model = CharCNN(CharCNNConfig.from_dict(ck["config"]))
        model.load_state_dict(ck["state_dict"])
        return cls(
            model=model,
            ood=MahalanobisOOD.from_dict(ck["ood"]),
            threshold=float(ck["threshold"]),
            novelty_threshold=float(ck["novelty_threshold"]),
            families=list(ck["families"]),
            version=str(ck.get("version", "unknown")),
            gate_auc=ck.get("gate_auc"),
            artifact_sha256=digest,
        )

    @torch.no_grad()
    def predict_batch(self, domains: list[str]) -> list[Verdict]:
        norm = [normalize(d) for d in domains]
        valid = [i for i, n in enumerate(norm) if n.ok]

        out: list[Verdict | None] = [None] * len(domains)
        for i, n in enumerate(norm):
            if not n.ok:
                out[i] = Verdict(
                    domain=domains[i],
                    core="",
                    verdict="unparseable",
                    dga_probability=0.0,
                    nearest_family=None,
                    novelty_score=None,
                    is_novel=False,
                    note=f"could not parse domain ({n.reason})",
                )

        if valid:
            cores = [norm[i].core for i in valid]
            x = encode_batch(cores, self.model.cfg.max_len)
            logit, _, emb = self.model(x)
            probs = torch.sigmoid(logit).numpy()
            e = emb.numpy()

            distances = self.ood.distances(e)
            closest = distances.argmin(axis=1)
            dists = distances[np.arange(len(closest)), closest]
            nearest = [self.ood.classes[index] for index in closest]

            note = None
            if not self.gate_is_trustworthy:
                note = (
                    "novelty gate is experimental, measured separation is near "
                    "chance, do not act on is_novel"
                )

            for k, i in enumerate(valid):
                is_dga = bool(probs[k] >= self.threshold)
                above_novelty_threshold = bool(dists[k] >= self.novelty_threshold)
                out[i] = Verdict(
                    domain=domains[i],
                    core=cores[k],
                    verdict="dga" if is_dga else "benign",
                    dga_probability=float(probs[k]),
                    nearest_family=nearest[k] if is_dga else None,
                    novelty_score=float(dists[k]),
                    is_novel=is_dga and above_novelty_threshold,
                    above_novelty_threshold=above_novelty_threshold,
                    note=note,
                )

        return [v for v in out if v is not None]

    def benchmark(self, domains: list[str], repeats: int = 50) -> dict:
        """Measure end-to-end inference latency for a given batch size."""
        self.predict_batch(domains[:1])
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self.predict_batch(domains)
            times.append((time.perf_counter() - t0) * 1000)
        a = np.array(times)
        return {
            "batch_size": len(domains),
            "repeats": repeats,
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "per_domain_p50_ms": float(np.percentile(a, 50) / max(len(domains), 1)),
        }
