from __future__ import annotations

import numpy as np


class SkipGramPairBuilder:
    """Build (center, context) pairs with a dynamic window, as in word2vec."""

    def __init__(self, window_size: int, rng: np.random.Generator) -> None:
        self.window_size = window_size
        self._rng = rng

    def build(self, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        if len(token_ids) < 2:
            raise ValueError("Need at least two tokens to form skip-gram pairs.")

        ids = np.asarray(token_ids, dtype=np.int32)
        n = ids.shape[0]
        windows = self._rng.integers(1, self.window_size + 1, size=n, dtype=np.int32)

        centers: list[int] = []
        contexts: list[int] = []
        for index, width in enumerate(windows):
            start = max(0, index - int(width))
            end = min(n, index + int(width) + 1)
            center = int(ids[index])
            for other in range(start, end):
                if other == index:
                    continue
                centers.append(center)
                contexts.append(int(ids[other]))

        return (
            np.asarray(centers, dtype=np.int64),
            np.asarray(contexts, dtype=np.int64),
        )
