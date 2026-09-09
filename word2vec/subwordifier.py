"""FastText character-n-gram hashing (Bojanowski et al., 2017).

Matches facebookresearch/fastText `Dictionary::hash` and `computeSubwords`:
FNV-1a with signed bytes, boundary markers `<` / `>`, n-grams of length
minn..maxn, then bucket = hash % num_buckets.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

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
        maxn: int | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if num_buckets < 1:
            raise ValueError("num_buckets must be >= 1")
        self.subword_embeddings = torch.nn.Embedding(num_buckets, embedding_dim, device=device)
        self.center_embeddings = torch.nn.Embedding(vocab_size, embedding_dim, device=device)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_buckets = num_buckets
        self.minn = minn
        self.maxn = maxn
        self.words = tuple(words) if words is not None else None
        self.word_to_bucket_ids_dict = self.precompute_bucket_hashes() if self.words is not None else None
        self.device = device

    def hash_ngram(self, ngram: str) -> int:
        return fasttext_hash(ngram) % self.num_buckets

    def hash_word(self, word: str) -> list[int]:
        """Bucket ids for every character n-gram of `word` (not the word row)."""
        return [self.hash_ngram(gram) for gram in character_ngrams(word, self.minn, self.maxn)]

    def precompute_bucket_hashes(self) -> list[list[int]]:
        if self.words is None:
            raise ValueError("id_to_word strings are required to precompute n-gram hashes")
        if len(self.words) != self.vocab_size:
            raise ValueError("len(words) must equal vocab_size")
        return {i: self.hash_word(word) for i, word in enumerate(self.words)}

    # Inefficient implementation: not vectorized
    def encode(self, center_index: torch.Tensor) -> torch.Tensor:
        """Word vector plus sum of n-gram bucket vectors (FastText training input)."""
        if self.word_to_bucket_ids_dict is None:
            raise ValueError("encode requires words= at init so bucket ids can be precomputed")
        device = self.center_embeddings.weight.device
        if center_index.ndim > 1:
            raise ValueError("encode expects a 0-D or 1-D tensor of word ids (skip-gram), not a CBOW bag")
        squeezed = center_index.ndim == 0
        ids = center_index.to(device).reshape(-1)

        center_vectors = self.center_embeddings(ids)
        ngram_sums = []
        for word_id in ids.tolist():
            bucket_ids = self.word_to_bucket_ids_dict[int(word_id)]
            if not bucket_ids:
                ngram_sums.append(
                    torch.zeros(self.embedding_dim, device=device, dtype=center_vectors.dtype)
                )
                continue
            buckets = torch.tensor(bucket_ids, dtype=torch.long, device=device)
            ngram_sums.append(self.subword_embeddings(buckets).sum(dim=0))
        out = center_vectors + torch.stack(ngram_sums, dim=0)
        # 0-D id → (dim,); 1-D ids → (batch, dim)
        return out.squeeze(0) if squeezed else out

    # TO DO: Vectorized implementation
    def encode_vectorized(self, center_index: torch.Tensor) -> torch.Tensor:
        pass
