from __future__ import annotations

import numpy as np


class SkipGramPairBuilder:
    """Dynamic windows for skip-gram pairs or CBOW context bags."""

    def __init__(self, window_size: int, rng: np.random.Generator) -> None:
        self.window_size = window_size
        self._rng = rng

    def _windows(self, token_ids: list[int]):
        if len(token_ids) < 2:
            raise ValueError("Need at least two tokens to form windows.")
        ids = np.asarray(token_ids, dtype=np.int32)
        n = ids.shape[0]
        widths = self._rng.integers(1, self.window_size + 1, size=n, dtype=np.int32)
        for index, width in enumerate(widths):
            start = max(0, index - int(width))
            end = min(n, index + int(width) + 1)
            bag = [int(ids[other]) for other in range(start, end) if other != index]
            yield int(ids[index]), bag

    def build(self, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        centers: list[int] = []
        contexts: list[int] = []
        for center, bag in self._windows(token_ids):
            for context in bag:
                centers.append(center)
                contexts.append(context)
        return (
            np.asarray(centers, dtype=np.int64),
            np.asarray(contexts, dtype=np.int64),
        )

    def build_cbow(self, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """One example per position: padded context bag predicting the center."""
        max_bag = 2 * self.window_size
        centers: list[int] = []
        bags: list[list[int]] = []
        for center, bag in self._windows(token_ids):
            if not bag:
                continue
            padded = [-1] * max_bag
            padded[: len(bag)] = bag
            centers.append(center)
            bags.append(padded)
        return (
            np.asarray(centers, dtype=np.int64),
            np.asarray(bags, dtype=np.int64),
        )
