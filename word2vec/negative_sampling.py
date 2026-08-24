from __future__ import annotations

import numpy as np

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
        """Return shape (batch_size, num_negatives), redrawing excluded positives."""

        negatives = self._rng.choice(self.table, size=(batch_size, num_negatives))
        for _ in range(10):
            collision = negatives == exclude[:, None]
            if not collision.any():
                break
            redraw = self._rng.choice(self.table, size=int(collision.sum()))
            negatives[collision] = redraw
        return negatives.astype(np.int64, copy=False)
