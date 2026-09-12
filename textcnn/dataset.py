from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

PAD_ID = 0
WORD_ID_OFFSET = 1


class SentenceDataset(Dataset):
    """Unpadded classified sentences. Batch collate pads to the widest example."""

    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        labels: Sequence[int],
    ) -> None:
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have the same length")
        if not sequences:
            raise ValueError("Need at least one labeled sentence.")
        self.sequences = [torch.as_tensor(list(ids), dtype=torch.long) for ids in sequences]
        self.labels = torch.as_tensor(list(labels), dtype=torch.long)
        if (self.labels < 0).any():
            raise ValueError("labels must be non-negative class ids")
        if any(int(row.numel()) < 1 for row in self.sequences):
            raise ValueError("every sequence must contain at least one token id")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": self.sequences[index],
            "label": self.labels[index],
        }


def pad_sentence_collate(
    batch: list[dict[str, torch.Tensor]],
    *,
    filter_sizes: tuple[int, ...],
    padding_idx: int = PAD_ID,
) -> dict[str, torch.Tensor]:
    """Pad token rows to `max(batch length, max(filter_sizes))` with `padding_idx`."""

    if not batch:
        raise ValueError("empty batch")
    if not filter_sizes:
        raise ValueError("filter_sizes must be non-empty")
    min_len = max(filter_sizes)
    max_len = max(min_len, max(int(item["tokens"].shape[0]) for item in batch))
    tokens = torch.full((len(batch), max_len), padding_idx, dtype=torch.long)
    for row, item in enumerate(batch):
        seq = item["tokens"]
        tokens[row, : seq.shape[0]] = seq
    labels = torch.stack([item["label"] for item in batch])
    return {"tokens": tokens, "label": labels}


def make_pad_collate(filter_sizes: tuple[int, ...], padding_idx: int = PAD_ID):
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return pad_sentence_collate(batch, filter_sizes=filter_sizes, padding_idx=padding_idx)

    return collate
