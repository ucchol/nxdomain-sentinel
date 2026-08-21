"""Character CNN over domain cores.

Deliberately small, roughly 150k parameters and well under 10 MB serialised.
Size is a serving decision rather than a modelling compromise. The container has
to cold start on a scale-to-zero platform, so a model that loads in milliseconds
and runs in single-digit milliseconds on one vCPU is worth more here than a
larger model that scores marginally better offline.

Parallel convolutions at widths 2 through 5 act as learned character n-gram
detectors, which suits the target. DGA labels are characterised by n-gram
statistics that differ from pronounceable registered names, and a width-4
filter is a direct analogue of the 4-gram features classical DGA detectors use.

The penultimate representation is exposed deliberately. It feeds the binary
head, the family head, and the out-of-distribution gate, so the OOD score is
computed in the same space the classifier actually decides in.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK = 0, 1
ALPHABET = string.ascii_lowercase + string.digits + "-."
STOI = {c: i + 2 for i, c in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET) + 2
MAX_LEN = 64


# Lookup table indexed by byte value. Encoding through a 256-entry table turns
# the inner loop into a numpy take rather than a per-character dict lookup,
# which matters because this runs over hundreds of thousands of rows at train
# time and on every request at serve time.
_BYTE_LUT = np.full(256, UNK, dtype=np.int64)
_BYTE_LUT[0] = PAD
for _ch, _i in STOI.items():
    _BYTE_LUT[ord(_ch)] = _i


def encode(core: str, max_len: int = MAX_LEN) -> list[int]:
    """Map a domain core to a fixed-length index sequence, right padded."""
    ids = [STOI.get(ch, UNK) for ch in core[:max_len]]
    return ids + [PAD] * (max_len - len(ids))


def encode_array(cores, max_len: int = MAX_LEN) -> np.ndarray:
    """Encode many cores into a preallocated (n, max_len) int64 array.

    Preallocating matters. Building a list of lists first allocates one Python
    object per character and was enough to exhaust memory on a small instance
    at a few hundred thousand rows.
    """
    n = len(cores)
    out = np.zeros((n, max_len), dtype=np.int64)
    for i, c in enumerate(cores):
        # str() guards the CSV fallback path, where domains such as nan.com and
        # inf.com come back as floats unless NA coercion is disabled.
        b = str(c).encode("ascii", "replace")[:max_len]
        if b:
            out[i, : len(b)] = _BYTE_LUT[np.frombuffer(b, dtype=np.uint8)]
    return out


def encode_batch(cores, max_len: int = MAX_LEN) -> torch.Tensor:
    return torch.from_numpy(encode_array(list(cores), max_len))


@dataclass
class CharCNNConfig:
    vocab_size: int = VOCAB_SIZE
    embed_dim: int = 32
    n_filters: int = 96
    kernel_sizes: tuple = (2, 3, 4, 5)
    hidden_dim: int = 128
    n_families: int = 25
    dropout: float = 0.3
    max_len: int = MAX_LEN

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kernel_sizes"] = list(self.kernel_sizes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CharCNNConfig":
        d = dict(d)
        d["kernel_sizes"] = tuple(d["kernel_sizes"])
        return cls(**d)


class CharCNN(nn.Module):
    """Two-headed character CNN.

    forward returns the binary logit, the family logits, and the penultimate
    embedding. Callers that only need a verdict can ignore the rest, but the
    embedding has to be reachable without a second pass because the OOD gate
    consumes it on every request.
    """

    def __init__(self, cfg: CharCNNConfig | None = None):
        super().__init__()
        self.cfg = cfg or CharCNNConfig()
        c = self.cfg

        self.embed = nn.Embedding(c.vocab_size, c.embed_dim, padding_idx=PAD)
        self.convs = nn.ModuleList(
            nn.Conv1d(c.embed_dim, c.n_filters, k) for k in c.kernel_sizes
        )
        conv_out = c.n_filters * len(c.kernel_sizes)

        self.dropout = nn.Dropout(c.dropout)
        self.proj = nn.Linear(conv_out, c.hidden_dim)
        self.norm = nn.LayerNorm(c.hidden_dim)

        self.binary_head = nn.Linear(c.hidden_dim, 1)
        self.family_head = nn.Linear(c.hidden_dim, c.n_families)

    def embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Produce the penultimate representation used by heads and the OOD gate."""
        e = self.embed(x).transpose(1, 2)
        feats = [F.relu(conv(e)).max(dim=2).values for conv in self.convs]
        h = self.dropout(torch.cat(feats, dim=1))
        return self.norm(F.relu(self.proj(h)))

    def forward(self, x: torch.Tensor):
        h = self.embed_tokens(x)
        return self.binary_head(h).squeeze(-1), self.family_head(h), h

    @torch.no_grad()
    def predict_proba(self, cores, batch_size: int = 512) -> torch.Tensor:
        self.eval()
        out = []
        for i in range(0, len(cores), batch_size):
            x = encode_batch(cores[i : i + batch_size], self.cfg.max_len)
            logit, _, _ = self(x)
            out.append(torch.sigmoid(logit))
        return torch.cat(out) if out else torch.empty(0)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
