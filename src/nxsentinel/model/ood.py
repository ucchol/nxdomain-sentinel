"""Out-of-distribution gate over the classifier's own embedding space.

A binary DGA score answers "does this look generated". It cannot answer "is this
a family I have seen before", and those are different questions for a defender.
A high-confidence hit on a known family is a signature match. A high-confidence
hit that sits far from every known family is a candidate novel campaign, which
is the alert worth waking someone for.

The gate fits one Gaussian per known family in the penultimate space, with a
single shared covariance pooled across families. That is Gaussian discriminant
analysis, and the resulting decision surfaces are class-conditional ellipsoids.
Shrinkage on the covariance keeps the precision matrix stable when a family has
few samples relative to the embedding dimension.

Scoring a point is the squared Mahalanobis distance to its nearest family
centroid. Small means "close to something known". Large means "nothing here
looks like this".

No OOD data is required to fit any of this, which is the property that makes it
usable. A gate that needs examples of the novel families it is meant to catch
would defeat its own purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MahalanobisOOD:
    """Fitted class-conditional Gaussian gate.

    Attributes:
        means: (n_classes, dim) family centroids in embedding space.
        precision: (dim, dim) shared inverse covariance.
        classes: family names aligned to the rows of ``means``.
        shrinkage: ridge weight applied to the pooled covariance.
    """

    means: np.ndarray
    precision: np.ndarray
    classes: list[str]
    shrinkage: float

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        families: list[str],
        shrinkage: float = 0.10,
    ) -> "MahalanobisOOD":
        """Fit centroids and a pooled, shrunk covariance from training data only."""
        embeddings = np.asarray(embeddings, dtype=np.float64)
        fams = np.asarray(families)
        classes = sorted(set(fams.tolist()))

        dim = embeddings.shape[1]
        means = np.zeros((len(classes), dim), dtype=np.float64)
        centered = np.empty_like(embeddings)

        for i, c in enumerate(classes):
            mask = fams == c
            mu = embeddings[mask].mean(axis=0)
            means[i] = mu
            centered[mask] = embeddings[mask] - mu

        # Pooling within-class scatter rather than fitting a covariance per
        # family keeps the estimate well conditioned even for small families.
        cov = centered.T @ centered / max(len(embeddings) - len(classes), 1)
        cov += shrinkage * np.trace(cov) / dim * np.eye(dim)

        return cls(
            means=means,
            precision=np.linalg.pinv(cov),
            classes=classes,
            shrinkage=shrinkage,
        )

    def distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance to every family centroid."""
        e = np.asarray(embeddings, dtype=np.float64)
        out = np.empty((len(e), len(self.classes)), dtype=np.float64)
        for i, mu in enumerate(self.means):
            d = e - mu
            out[:, i] = np.einsum("ij,jk,ik->i", d, self.precision, d)
        return out

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """OOD score, higher means further from every known family."""
        return self.distances(embeddings).min(axis=1)

    def nearest_family(self, embeddings: np.ndarray) -> list[str]:
        idx = self.distances(embeddings).argmin(axis=1)
        return [self.classes[i] for i in idx]

    def to_dict(self) -> dict:
        return {
            "means": self.means.tolist(),
            "precision": self.precision.tolist(),
            "classes": list(self.classes),
            "shrinkage": self.shrinkage,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MahalanobisOOD":
        return cls(
            means=np.asarray(d["means"], dtype=np.float64),
            precision=np.asarray(d["precision"], dtype=np.float64),
            classes=list(d["classes"]),
            shrinkage=float(d["shrinkage"]),
        )


@torch.no_grad()
def collect_embeddings(model, cores, batch_size: int = 512) -> np.ndarray:
    """Run the encoder over ``cores`` and return penultimate embeddings."""
    from .charcnn import encode_batch

    model.eval()
    chunks = []
    for i in range(0, len(cores), batch_size):
        x = encode_batch(list(cores[i : i + batch_size]), model.cfg.max_len)
        chunks.append(model.embed_tokens(x).numpy())
    return np.concatenate(chunks) if chunks else np.empty((0, model.cfg.hidden_dim))
