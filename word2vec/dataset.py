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


def word_ngrams(token_ids: Sequence[int], ngram_size: int) -> np.ndarray:
    """Consecutive word-id n-grams with shape `(num_ngrams, ngram_size)`."""
    if ngram_size < 2:
        raise ValueError("ngram_size must be >= 2")
    ids = np.asarray(token_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("token_ids must be 1-D")
    if ids.shape[0] < ngram_size:
        return np.empty((0, ngram_size), dtype=np.int64)
    return np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(ids, ngram_size))


class FastTextDataset(Dataset):
    """Unpadded classified documents. Batch collate pads to the widest example."""

    def __init__(
        self,
        features: Sequence[np.ndarray],
        weights: Sequence[np.ndarray],
        ngrams: Sequence[np.ndarray],
        token_counts: Sequence[int],
        labels: Sequence[int],
    ) -> None:
        if not (len(features) == len(weights) == len(ngrams) == len(token_counts) == len(labels)):
            raise ValueError("features, weights, ngrams, token_counts, and labels must align")
        if not features:
            raise ValueError("Need at least one FastText example")
        self.features = [torch.as_tensor(row, dtype=torch.long) for row in features]
        self.weights = [torch.as_tensor(row, dtype=torch.float32) for row in weights]
        self.ngrams = [torch.as_tensor(np.array(row, copy=True), dtype=torch.long) for row in ngrams]
        self.token_counts = torch.as_tensor(token_counts, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        for feature_row, weight_row, ngram_row in zip(self.features, self.weights, self.ngrams):
            if feature_row.ndim != 1 or weight_row.ndim != 1:
                raise ValueError("each document's features and weights must be 1-D")
            if feature_row.shape != weight_row.shape:
                raise ValueError("features and weights must have the same length")
            if ngram_row.ndim != 2:
                raise ValueError("each document's ngrams must be 2-D (num_ngrams, ngram_size)")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "weights": self.weights[index],
            "ngrams": self.ngrams[index],
            "token_count": self.token_counts[index],
            "label": self.labels[index],
        }


def pad_fasttext_collate(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Pad features and n-grams to the widest example in this batch only."""
    max_features = max(item["features"].shape[0] for item in batch)
    ngram_size = int(batch[0]["ngrams"].shape[1])
    max_ngrams = max(item["ngrams"].shape[0] for item in batch)
    batch_size = len(batch)
    features = torch.full((batch_size, max_features), -1, dtype=torch.long)
    weights = torch.zeros((batch_size, max_features), dtype=torch.float32)
    ngrams = torch.full((batch_size, max_ngrams, ngram_size), -1, dtype=torch.long)
    for index, item in enumerate(batch):
        feature_width = int(item["features"].shape[0])
        features[index, :feature_width] = item["features"]
        weights[index, :feature_width] = item["weights"]
        gram_count = int(item["ngrams"].shape[0])
        if gram_count:
            ngrams[index, :gram_count] = item["ngrams"]
    return {
        "features": features,
        "weights": weights,
        "ngrams": ngrams,
        "token_count": torch.stack([item["token_count"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
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
