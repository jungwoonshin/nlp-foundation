from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from word2vec.subword import Subwordifier
from word2vec.vocab import Vocab


class BOW_FastText(nn.Module):
    """Skip-gram NEG with FastText inputs: center is word + hashed n-grams."""

    def __init__(
        self,
        embedding_dim: int,
        vocab: Vocab,
        num_buckets: int = 2_000_000,
    ) -> None:
        super().__init__()
        vocab_size = len(vocab)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.subwordifier = Subwordifier(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            num_buckets=num_buckets,
            words=vocab.id_to_word,
            minn=3,
            maxn=6,
        )

    def forward(
        self,
        center_index: torch.Tensor,
        target_index: torch.Tensor,
        negative_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return mean BCE. `center_index` is a 1-D batch of skip-gram input word ids."""
        input_vectors = self.subwordifier.encode(center_index)
        context_vectors = self.context_embedding(target_index)
        negative_vectors = self.context_embedding(negative_indices)

        positive_logits = (context_vectors * input_vectors).sum(dim=-1)
        negative_logits = (negative_vectors * input_vectors.unsqueeze(1)).sum(dim=-1)

        pos_bce = F.binary_cross_entropy_with_logits(
            positive_logits, torch.ones_like(positive_logits), reduction="none"
        )
        neg_bce = F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits), reduction="none"
        )
        return (pos_bce + neg_bce.sum(dim=1)).mean()
