from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from word2vec.embed import lookup_weighted
from word2vec.fasttext.hashing import hash_word_ngram
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
        self.input_embedding = nn.Embedding(self.vocab_size + num_buckets, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def ngram_ids(self, ngrams: torch.Tensor) -> torch.Tensor:
        """Map word-id n-grams to rows `V + (hash % buckets)` in the shared table."""
        if ngrams.ndim != 3:
            raise ValueError("ngrams must be 3-D (batch, num_ngrams, ngram_size)")
        batch_size, num_ngrams, _ = ngrams.shape
        ids = torch.full(
            (batch_size, num_ngrams),
            -1,
            dtype=torch.long,
            device=ngrams.device,
        )
        words = self.id_to_word
        offset = self.vocab_size
        for row, gram_row in enumerate(ngrams.tolist()):
            for col, gram in enumerate(gram_row):
                tokens = [words[int(token_id)] for token_id in gram if int(token_id) >= 0]
                if not tokens:
                    continue
                ids[row, col] = offset + hash_word_ngram(tokens) % self.num_buckets
        return ids

    def encode(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        ngrams: torch.Tensor,
        token_count: torch.Tensor,
    ) -> torch.Tensor:
        """Average every feature in the bag: words and hashed n-grams from A."""
        word_mean = lookup_weighted(self.input_embedding, features, weights)
        counts = token_count.to(dtype=word_mean.dtype).reshape(-1, 1)
        word_sum = word_mean * counts

        ngram_ids = self.ngram_ids(ngrams)
        present = ngram_ids >= 0
        ngram_vectors = self.input_embedding(ngram_ids.clamp(min=0))
        ngram_mask = present.unsqueeze(-1).to(dtype=ngram_vectors.dtype)
        ngram_sum = (ngram_vectors * ngram_mask).sum(dim=1)
        ngram_count = ngram_mask.sum(dim=1)

        return (word_sum + ngram_sum) / (counts + ngram_count).clamp(min=1.0)

    def forward(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        ngrams: torch.Tensor,
        token_count: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encode(features, weights, ngrams, token_count)
        logits = self.classifier(hidden)
        return F.cross_entropy(logits, labels)
