from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from word2vec.negative_sampling import NegativeSampler


class SkipGramDataset(Dataset):
    """Positive skip-gram pairs as integer tensors."""

    def __init__(self, centers: np.ndarray, contexts: np.ndarray) -> None:
        if centers.shape != contexts.shape:
            raise ValueError("centers and contexts must have the same shape")
        self.centers = torch.as_tensor(centers, dtype=torch.long)
        self.contexts = torch.as_tensor(contexts, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.centers.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "center": self.centers[index],
            "context": self.contexts[index],
        }


def make_negative_collate(
    sampler: NegativeSampler,
    num_negatives: int,
) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        centers = torch.stack([item["center"] for item in batch])
        contexts = torch.stack([item["context"] for item in batch])
        negatives = sampler.sample(
            batch_size=len(batch),
            num_negatives=num_negatives,
            exclude=contexts.numpy(),
        )
        return {
            "center": centers,
            "context": contexts,
            "negatives": torch.from_numpy(negatives),
        }

    return collate
