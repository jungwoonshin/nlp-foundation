from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from word2vec.vocab import Vocab


class NegativeSampler:
    """Draw negative word ids from a unigram^{power} noise distribution."""

    def __init__(
        self,
        vocab: Vocab,
        power: float,
        table_size: int,
        rng: np.random.Generator,
    ) -> None:
        weights = np.asarray(vocab.counts, dtype=np.float64) ** power
        probabilities = weights / weights.sum()
        self.table = rng.choice(len(vocab), size=table_size, replace=True, p=probabilities)
        self._rng = rng

    def sample(self, batch_size: int, num_negatives: int, exclude: np.ndarray) -> np.ndarray:
        """Return shape (batch_size, num_negatives), redrawing excluded ids.

        `exclude` is (batch,) or (batch, n): typically the true context and the
        center word, so a negative is neither the positive nor the input word.
        """
        blocked = np.asarray(exclude)
        if blocked.ndim == 1:
            blocked = blocked[:, None]
        negatives = self._rng.choice(self.table, size=(batch_size, num_negatives))
        for _ in range(10):
            collision = (negatives[..., None] == blocked[:, None, :]).any(axis=-1)
            if not collision.any():
                break
            redraw = self._rng.choice(self.table, size=int(collision.sum()))
            negatives[collision] = redraw
        return negatives.astype(np.int64, copy=False)


class NegativeSampling(nn.Module):
    """Skip-gram with negative sampling (two embedding banks + BCE).

    `forward(center_index, target_index, negative_indices)` looks up the center
    (input) vector and context (output) vectors for the true neighbor and K
    noise words, then scores each pair by a dot product.
    """

    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(
        self,
        center_index: torch.Tensor,
        target_index: torch.Tensor,
        negative_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return mean binary cross entropy for skip-gram pairs given as word ids."""
        center_vectors = self.center_embeddings(center_index)
        context_vectors = self.context_embedding(target_index)
        negative_vectors = self.context_embedding(negative_indices)

        positive_logits = (context_vectors * center_vectors).sum(dim=-1)
        negative_logits = (negative_vectors * center_vectors.unsqueeze(1)).sum(dim=-1)

        pos_bce = F.binary_cross_entropy_with_logits(
            positive_logits, torch.ones_like(positive_logits), reduction="none"
        )
        neg_bce = F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits), reduction="none"
        )
        return (pos_bce + neg_bce.sum(dim=1)).mean()
