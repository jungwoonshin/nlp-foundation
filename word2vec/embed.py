from __future__ import annotations

import torch
from torch import nn


def lookup_mean(embedding: nn.Embedding, token_ids: torch.Tensor) -> torch.Tensor:
    """Map skip-gram ids `(batch,)` or CBOW bags `(batch, window)` to `(batch, dim)`.

    Bag positions with id < 0 are padding and are left out of the mean.
    """
    if token_ids.ndim == 1:
        return embedding(token_ids)
    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must be 1-D or 2-D, got {tuple(token_ids.shape)}")
    present = token_ids >= 0
    vectors = embedding(token_ids.clamp(min=0))
    weights = present.unsqueeze(-1).to(dtype=vectors.dtype)
    counted = weights.sum(dim=1).clamp(min=1.0)
    return (vectors * weights).sum(dim=1) / counted


def lookup_weighted(
    embedding: nn.Embedding,
    token_ids: torch.Tensor,
    feature_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted bag: `sum(embed(id) * freq)` for ids >= 0. Weights should sum to 1."""
    if token_ids.ndim != 2 or feature_weights.shape != token_ids.shape:
        raise ValueError("token_ids and feature_weights must be 2-D with the same shape")
    present = token_ids >= 0
    vectors = embedding(token_ids.clamp(min=0))
    weights = feature_weights.to(dtype=vectors.dtype) * present.to(dtype=vectors.dtype)
    return (vectors * weights.unsqueeze(-1)).sum(dim=1)
