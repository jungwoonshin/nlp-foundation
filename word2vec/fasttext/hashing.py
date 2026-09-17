"""Word n-gram hashing from facebookresearch/fastText `Dictionary::addNgrams`."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from word2vec.subword import fasttext_hash

# dictionary.cc: h = h * 116049371 + hashes[j]
_WORD_NGRAM_PRIME = 116049371


def hash_word_ngram(words: Sequence[str]) -> int:
    """64-bit FastText hash of an ordered word n-gram."""
    if not words:
        raise ValueError("Need at least one word to hash an n-gram.")
    h = fasttext_hash(words[0])
    for word in words[1:]:
        h = (h * _WORD_NGRAM_PRIME + fasttext_hash(word)) & 0xFFFFFFFFFFFFFFFF
    return h


def hashed_token_ngrams(
    tokens: Sequence[str],
    ngram_size: int,
    vocab_size: int,
    num_buckets: int,
) -> np.ndarray:
    """Embedding-table rows `V + (hash % buckets)` for consecutive word n-grams."""
    if ngram_size < 2:
        raise ValueError("ngram_size must be >= 2")
    if len(tokens) < ngram_size:
        return np.empty((0,), dtype=np.int64)
    out = np.empty((len(tokens) - ngram_size + 1,), dtype=np.int64)
    for index in range(out.shape[0]):
        out[index] = vocab_size + hash_word_ngram(tokens[index : index + ngram_size]) % num_buckets
    return out
