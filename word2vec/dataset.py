from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from word2vec.negative_sampling import NegativeSampler


class SkipGramDataset(Dataset):
    """Skip-gram pairs or CBOW bags as integer tensors.

    Skip-gram: `contexts` is 1-D (one neighbor). CBOW: `contexts` is 2-D
    (padded bag, -1 = pad) predicting `centers`.
    """

    def __init__(self, centers: np.ndarray, contexts: np.ndarray) -> None:
        if centers.ndim != 1:
            raise ValueError("centers must be 1-D")
        if contexts.ndim == 1:
            if centers.shape != contexts.shape:
                raise ValueError("skip-gram centers and contexts must have the same shape")
        elif contexts.ndim == 2:
            if contexts.shape[0] != centers.shape[0]:
                raise ValueError("CBOW bags must align with centers")
        else:
            raise ValueError("contexts must be 1-D or 2-D")
        self.centers = torch.as_tensor(centers, dtype=torch.long)
        self.contexts = torch.as_tensor(contexts, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.centers.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "center": self.centers[index],
            "context": self.contexts[index],
        }


def normalized_feature_frequency(token_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Unique feature ids and counts / document length (sum of weights is 1)."""
    if not token_ids:
        raise ValueError("Need at least one token to form FastText features.")
    counts = Counter(int(token_id) for token_id in token_ids)
    total = float(sum(counts.values()))
    features = np.fromiter(counts, dtype=np.int64, count=len(counts))
    weights = np.fromiter(
        (counts[int(feature)] / total for feature in features),
        dtype=np.float32,
        count=len(counts),
    )
    return features, weights


class FastTextDataset(Dataset):
    """One classified document: normalized feature frequencies -> label id."""

    def __init__(
        self,
        features: np.ndarray,
        weights: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        if features.ndim != 2 or weights.ndim != 2 or labels.ndim != 1:
            raise ValueError("features and weights must be 2-D and labels 1-D")
        if features.shape != weights.shape or features.shape[0] != labels.shape[0]:
            raise ValueError("features, weights, and labels must align")
        self.features = torch.as_tensor(features, dtype=torch.long)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "weights": self.weights[index],
            "label": self.labels[index],
        }


def make_negative_collate(
    sampler: NegativeSampler,
    num_negatives: int,
) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        centers = torch.stack([item["center"] for item in batch])
        contexts = torch.stack([item["context"] for item in batch])
        if contexts.ndim == 1:
            exclude = np.stack([contexts.numpy(), centers.numpy()], axis=1)
        else:
            exclude = np.concatenate(
                [centers.numpy()[:, None], contexts.numpy()],
                axis=1,
            )
        negatives = sampler.sample(
            batch_size=len(batch),
            num_negatives=num_negatives,
            exclude=exclude,
        )
        return {
            "center": centers,
            "context": contexts,
            "negatives": torch.from_numpy(negatives),
        }

    return collate
