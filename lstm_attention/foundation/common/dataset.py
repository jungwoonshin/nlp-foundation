from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
FIRST_TOKEN = 3


class ReverseDataset(Dataset):
    """Source token rows; target is the reversed source with BOS/EOS."""

    def __init__(self, sequences: Sequence[Sequence[int]]) -> None:
        if not sequences:
            raise ValueError("Need at least one source sequence.")
        self.sources = [list(ids) for ids in sequences]
        for row in self.sources:
            if not row:
                raise ValueError("every sequence must contain at least one token id")
            if any(int(token) < FIRST_TOKEN for token in row):
                raise ValueError("content tokens must be >= FIRST_TOKEN")

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        src = self.sources[index]
        tgt = list(reversed(src))
        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt_in": torch.tensor([BOS_ID, *tgt], dtype=torch.long),
            "tgt_out": torch.tensor([*tgt, EOS_ID], dtype=torch.long),
        }


def pad_collate(
    batch: list[dict[str, torch.Tensor]],
    padding_idx: int = PAD_ID,
) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("empty batch")
    max_src = max(int(item["src"].shape[0]) for item in batch)
    max_tgt = max(int(item["tgt_in"].shape[0]) for item in batch)
    src = torch.full((len(batch), max_src), padding_idx, dtype=torch.long)
    tgt_in = torch.full((len(batch), max_tgt), padding_idx, dtype=torch.long)
    tgt_out = torch.full((len(batch), max_tgt), padding_idx, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for row, item in enumerate(batch):
        src_row = item["src"]
        src[row, : src_row.shape[0]] = src_row
        lengths[row] = src_row.shape[0]
        tgt_in[row, : item["tgt_in"].shape[0]] = item["tgt_in"]
        tgt_out[row, : item["tgt_out"].shape[0]] = item["tgt_out"]
    return {
        "src": src,
        "src_lengths": lengths,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
    }


@dataclass
class ProcessedReverse:
    dataset: ReverseDataset
    vocab_size: int

    def dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=pad_collate,
        )


def make_reverse_dataset(
    num_examples: int,
    vocab_size: int,
    min_len: int = 2,
    max_len: int = 5,
    seed: int = 0,
) -> ProcessedReverse:
    if num_examples < 1:
        raise ValueError("num_examples must be >= 1")
    if vocab_size <= FIRST_TOKEN:
        raise ValueError("vocab_size must include PAD, BOS, EOS, and at least one content token")
    if min_len < 1:
        raise ValueError("min_len must be >= 1")
    if max_len < min_len:
        raise ValueError("max_len must be >= min_len")
    rng = torch.Generator()
    rng.manual_seed(seed)
    sequences: list[list[int]] = []
    for _ in range(num_examples):
        length = int(torch.randint(min_len, max_len + 1, (1,), generator=rng).item())
        seq = torch.randint(FIRST_TOKEN, vocab_size, (length,), generator=rng).tolist()
        sequences.append(seq)
    return ProcessedReverse(ReverseDataset(sequences), vocab_size=vocab_size)


def smoke_reverse() -> ProcessedReverse:
    return make_reverse_dataset(
        num_examples=4,
        vocab_size=8,
        min_len=2,
        max_len=3,
        seed=0,
    )


def trim_prediction(seq: torch.Tensor, *, pad_id: int = PAD_ID, eos_id: int = EOS_ID) -> list[int]:
    """Tokens up to and including the first EOS, ignoring trailing PAD."""
    out: list[int] = []
    for token in seq.tolist():
        if token == pad_id:
            break
        out.append(int(token))
        if token == eos_id:
            break
    return out
