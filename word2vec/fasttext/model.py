from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from word2vec.embed import lookup_weighted
from word2vec.vocab import Vocab


class BOW_FastText(nn.Module):
    """Supervised FastText: one input table A for words and hashed n-grams."""

    def __init__(
        self,
        embedding_dim: int,
        vocab: Vocab,
        num_classes: int,
        num_buckets: int = 10_000_000,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if num_buckets < 1:
            raise ValueError("num_buckets must be >= 1")
        self.id_to_word = vocab.id_to_word
        self.vocab_size = len(vocab)
        self.num_buckets = num_buckets
        # Official input_: rows [0, V) are words, [V, V + buckets) are n-grams.
        # sparse=True so SGD does not materialize a dense (V+buckets) gradient.
        self.input_embedding = nn.Embedding(
            self.vocab_size + num_buckets, embedding_dim, sparse=True
        )
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        self.init_like_fasttext()

    def init_like_fasttext(self) -> None:
        """Match official FastText: input uniform ±1/dim, output zeros, no bias."""
        dim = self.input_embedding.embedding_dim
        bound = 1.0 / dim
        nn.init.uniform_(self.input_embedding.weight, -bound, bound)
        nn.init.zeros_(self.classifier.weight)

    def encode(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        ngrams: torch.Tensor,
    ) -> torch.Tensor:
        """Sum freq * vector for words and n-grams, then divide by total feature count."""
        present_words = features >= 0
        word_freq = weights.to(dtype=self.input_embedding.weight.dtype) * present_words
        word_sum = lookup_weighted(self.input_embedding, features, word_freq)
        word_count = word_freq.sum(dim=1, keepdim=True)

        present_ngrams = ngrams >= 0
        ngram_freq = present_ngrams.to(dtype=word_sum.dtype)
        ngram_sum = lookup_weighted(self.input_embedding, ngrams, ngram_freq)
        ngram_count = ngram_freq.sum(dim=1, keepdim=True)

        count = (word_count + ngram_count).clamp(min=1.0)
        return (word_sum + ngram_sum) / count

    def forward(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        ngrams: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encode(features, weights, ngrams)
        logits = self.classifier(hidden)
        return F.cross_entropy(logits, labels)
