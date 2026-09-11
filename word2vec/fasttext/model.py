from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from word2vec.embed import lookup_weighted
from word2vec.vocab import Vocab


class BOW_FastText(nn.Module):
    """Supervised FastText: normalized bag-of-words features predict a class."""

    def __init__(
        self,
        embedding_dim: int,
        vocab: Vocab,
        num_classes: int,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.input_embedding = nn.Embedding(len(vocab), embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        hidden = lookup_weighted(self.input_embedding, features, weights)
        logits = self.classifier(hidden)
        return F.cross_entropy(logits, labels)
