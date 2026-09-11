"""Word n-gram hashing from facebookresearch/fastText `Dictionary::addNgrams`."""

from __future__ import annotations

from collections.abc import Sequence

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
