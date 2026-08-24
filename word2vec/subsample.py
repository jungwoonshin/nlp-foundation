from __future__ import annotations

import numpy as np

from word2vec.vocab import Vocab


class FrequentWordSubsampler:
    """Drop frequent tokens with the Mikolov et al. (2013) keep probability.

    P(keep w) = min(1, sqrt(t / f(w)) + t / f(w))
    where f(w) is the corpus frequency of w.
    """

    def __init__(self, vocab: Vocab, threshold: float, rng: np.random.Generator) -> None:
        total = vocab.total_tokens()
        frequencies = np.asarray(vocab.counts, dtype=np.float64) / total
        keep = np.sqrt(threshold / frequencies) + threshold / frequencies
        self._keep_prob = np.minimum(1.0, keep)
        self._rng = rng

    def apply(self, token_ids: list[int]) -> list[int]:
        ids = np.asarray(token_ids, dtype=np.int32)
        draws = self._rng.random(len(ids))
        kept = ids[draws < self._keep_prob[ids]]
        return kept.tolist()
