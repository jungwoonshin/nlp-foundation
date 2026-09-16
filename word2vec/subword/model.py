"""FastText character-n-gram hashing (Bojanowski et al., 2017).

Matches facebookresearch/fastText `Dictionary::hash` and `computeSubwords`:
FNV-1a with signed bytes, boundary markers `<` / `>`, n-grams of length
minn..maxn, then bucket = hash % num_buckets.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from word2vec.vocab import Vocab

# Official FNV-1a constants from src/dictionary.cc
_FNV_OFFSET = 2166136261
_FNV_PRIME = 16777619
_BOW = "<"
_EOW = ">"


def fasttext_hash(text: str) -> int:
    """32-bit FastText hash of `text` (UTF-8 bytes, sign-extended like `int8_t`)."""
    h = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        signed = byte - 256 if byte >= 128 else byte
        h ^= signed & 0xFFFFFFFF
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    return h


def _is_utf8_continuation(byte: int) -> bool:
    return (byte & 0xC0) == 0x80


def character_ngrams(word: str, minn: int, maxn: int) -> list[str]:
    """n-grams of `<word>` as in `Dictionary::computeSubwords` (UTF-8 aware)."""
    wrapped = f"{_BOW}{word}{_EOW}"
    data = wrapped.encode("utf-8")
    grams: list[str] = []
    size = len(data)
    for start, first in enumerate(data):
        if _is_utf8_continuation(first):
            continue
        end = start
        n = 1
        while end < size and n <= maxn:
            end += 1
            while end < size and _is_utf8_continuation(data[end]):
                end += 1
            if n >= minn and not (n == 1 and (start == 0 or end == size)):
                grams.append(data[start:end].decode("utf-8"))
            n += 1
    return grams


class Subwordifier(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        num_buckets: int,
        words: Sequence[str] | None = None,
        minn: int = 3,
        maxn: int = 6,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if num_buckets < 1:
            raise ValueError("num_buckets must be >= 1")
        if minn < 1 or maxn < minn:
            raise ValueError("require 1 <= minn <= maxn")
        self.subword_embeddings = torch.nn.Embedding(num_buckets, embedding_dim, device=device)
        self.center_embeddings = torch.nn.Embedding(vocab_size, embedding_dim, device=device)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_buckets = num_buckets
        self.minn = minn
        self.maxn = maxn
        self.words = tuple(words) if words is not None else None
        self.device = device
        if self.words is not None:
            if len(self.words) != vocab_size:
                raise ValueError("len(words) must equal vocab_size")
            self._register_bucket_table()
        else:
            self.register_buffer("bucket_table", torch.empty(0, 0, dtype=torch.long), persistent=False)

    def hash_ngram(self, ngram: str) -> int:
        return fasttext_hash(ngram) % self.num_buckets

    def hash_word(self, word: str) -> list[int]:
        """Bucket ids for every character n-gram of `word` (not the word row)."""
        return [self.hash_ngram(gram) for gram in character_ngrams(word, self.minn, self.maxn)]

    def precompute_bucket_hashes(self) -> dict[int, list[int]]:
        if self.words is None:
            raise ValueError("id_to_word strings are required to precompute n-gram hashes")
        return {i: self.hash_word(word) for i, word in enumerate(self.words)}

    def _register_bucket_table(self) -> None:
        """Padded `(V, max_n)` bucket ids; `-1` is a masked position."""
        rows = [self.hash_word(word) for word in self.words or ()]
        max_n = max((len(row) for row in rows), default=0)
        table = torch.full((self.vocab_size, max_n), -1, dtype=torch.long)
        for index, row in enumerate(rows):
            if row:
                table[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        self.register_buffer("bucket_table", table, persistent=False)

    def encode(self, center_index: torch.Tensor) -> torch.Tensor:
        """Average of the word row and its hashed n-gram rows (FastText `computeHidden`)."""
        if self.bucket_table.numel() == 0 and self.words is None:
            raise ValueError("encode requires words= at init so bucket ids can be precomputed")
        device = self.center_embeddings.weight.device
        if center_index.ndim > 1:
            raise ValueError("encode expects a 0-D or 1-D tensor of word ids (skip-gram), not a CBOW bag")
        squeezed = center_index.ndim == 0
        ids = center_index.to(device=device, dtype=torch.long).reshape(-1)
        word_vectors = self.center_embeddings(ids)
        if self.bucket_table.ndim != 2 or self.bucket_table.shape[1] == 0:
            return word_vectors.squeeze(0) if squeezed else word_vectors
        buckets = self.bucket_table[ids]
        present = buckets >= 0
        ngram_vectors = self.subword_embeddings(buckets.clamp(min=0))
        ngram_sum = (ngram_vectors * present.unsqueeze(-1).to(dtype=ngram_vectors.dtype)).sum(dim=1)
        denom = present.sum(dim=1, keepdim=True).to(dtype=word_vectors.dtype) + 1.0
        out = (word_vectors + ngram_sum) / denom
        return out.squeeze(0) if squeezed else out

    def encode_vectorized(self, center_index: torch.Tensor) -> torch.Tensor:
        return self.encode(center_index)


class SubwordNegativeSampling(nn.Module):
    """Skip-gram NEG with FastText inputs: center is word + hashed n-grams."""

    def __init__(
        self,
        embedding_dim: int,
        vocab: Vocab,
        num_buckets: int = 2_000_000,
        minn: int = 3,
        maxn: int = 6,
    ) -> None:
        super().__init__()
        vocab_size = len(vocab)
        self.word_to_id = vocab.word_to_id
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.subwordifier = Subwordifier(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            num_buckets=num_buckets,
            words=vocab.id_to_word,
            minn=minn,
            maxn=maxn,
        )
        self.init_like_fasttext()

    def init_like_fasttext(self) -> None:
        """Match official FastText: input uniform ±1/dim, output zeros."""
        dim = self.subwordifier.embedding_dim
        bound = 1.0 / dim
        nn.init.uniform_(self.subwordifier.center_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.subwordifier.subword_embeddings.weight, -bound, bound)
        nn.init.zeros_(self.context_embedding.weight)

    @torch.no_grad()
    def compose(self, word: str) -> torch.Tensor:
        """SISG vector: mean of the word row (if in-vocab) and character n-grams."""
        sub = self.subwordifier
        index = self.word_to_id.get(word)
        if index is None:
            index = self.word_to_id.get(word.lower())
            word = word.lower()
        gram_ids = sub.hash_word(sub.words[index] if index is not None else word)
        parts: list[torch.Tensor] = []
        count = 0
        if index is not None:
            parts.append(sub.center_embeddings.weight[index])
            count += 1
        if gram_ids:
            buckets = torch.tensor(gram_ids, dtype=torch.long, device=sub.subword_embeddings.weight.device)
            parts.append(sub.subword_embeddings(buckets).sum(dim=0))
            count += len(gram_ids)
        if not parts:
            return torch.zeros(sub.embedding_dim, device=sub.center_embeddings.weight.device)
        stacked = parts[0] if len(parts) == 1 else parts[0] + parts[1]
        return stacked / max(count, 1)

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
