"""Fit one split and score it.

The family head is trained alongside the binary head on purpose. Forcing the
network to separate the 20 training families pushes the embedding to encode
which generator produced a string, not merely whether one did. That structure is
what the OOD gate later measures distance in, so a binary-only model would leave
the gate with a much weaker space to work from.

Benign rows carry no family label and are masked out of the family loss rather
than assigned a placeholder class.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..model.charcnn import CharCNN, CharCNNConfig, encode_batch
from ..model.ood import MahalanobisOOD, collect_embeddings


@dataclass
class TrainConfig:
    epochs: int = 6
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 1e-5
    family_loss_weight: float = 0.3
    seed: int = 0
    max_train_rows: int | None = None


@torch.no_grad()
def _forward_logits(model, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Binary logits for a whole tensor, evaluated in batches.

    A single forward pass over a full validation set materialises the
    convolution activations for every row at once, which is gigabytes at this
    row count. Batching keeps peak memory flat and costs nothing.
    """
    model.eval()
    out = [model(x[i : i + batch_size])[0] for i in range(0, len(x), batch_size)]
    return torch.cat(out) if out else torch.empty(0)


def _tensors(df, family_to_idx: dict[str, int], max_len: int):
    x = encode_batch(df["core"].tolist(), max_len)
    y = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)
    fam = torch.tensor(
        [family_to_idx.get(f, -1) for f in df["family"]], dtype=torch.long
    )
    return x, y, fam


def train_one_split(split, cfg: TrainConfig | None = None, verbose: bool = True):
    """Train a model on ``split.train`` and fit the OOD gate on the same rows."""
    cfg = cfg or TrainConfig()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_df = split.train
    if cfg.max_train_rows and len(train_df) > cfg.max_train_rows:
        train_df = train_df.sample(
            n=cfg.max_train_rows, random_state=cfg.seed
        ).reset_index(drop=True)

    fams = sorted(train_df.loc[train_df.label == 1, "family"].unique())
    family_to_idx = {f: i for i, f in enumerate(fams)}

    model = CharCNN(CharCNNConfig(n_families=max(len(fams), 1)))
    max_len = model.cfg.max_len

    xtr, ytr, ftr = _tensors(train_df, family_to_idx, max_len)
    xva, yva, fva = _tensors(split.val, family_to_idx, max_len)

    loader = DataLoader(
        TensorDataset(xtr, ytr, ftr),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(loader)
    )
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    history = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for xb, yb, fb in loader:
            opt.zero_grad()
            logit, fam_logits, _ = model(xb)

            loss = bce(logit, yb)
            mask = fb >= 0
            if mask.any():
                loss = loss + cfg.family_loss_weight * ce(fam_logits[mask], fb[mask])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * len(xb)

        vlogit = _forward_logits(model, xva, cfg.batch_size)
        vloss = bce(vlogit, yva).item()
        vacc = ((vlogit > 0).float() == yva).float().mean().item()

        rec = {
            "epoch": epoch,
            "train_loss": running / len(xtr),
            "val_loss": vloss,
            "val_acc": vacc,
        }
        history.append(rec)
        if verbose:
            print(
                f"  epoch {epoch}  train {rec['train_loss']:.4f}  "
                f"val {vloss:.4f}  val_acc {vacc:.4f}  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    # The gate is fitted on DGA training rows only. Benign has no family label,
    # and mixing it in would blur the very centroids the gate measures against.
    dga_train = train_df[train_df.label == 1]
    emb = collect_embeddings(model, dga_train["core"].tolist())
    ood = MahalanobisOOD.fit(emb, dga_train["family"].tolist())

    return model, ood, {"history": history, "families": fams, "seconds": time.time() - t0}
