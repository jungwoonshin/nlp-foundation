from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from seq_to_seq.attention.config import LuongConfig
from seq_to_seq.attention.data.corpus import (
    filter_by_length,
    load_parallel_lines,
    maybe_reverse,
    smoke_token_pairs,
)
from seq_to_seq.attention.data.vocab import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    Vocab,
    build_vocab,
)


class ParallelDataset(Dataset):
    """Integer source/target rows with BOS/EOS teacher-forcing frames."""

    def __init__(
        self,
        sources: Sequence[Sequence[int]],
        targets: Sequence[Sequence[int]],
    ) -> None:
        if not sources:
            raise ValueError("Need at least one parallel pair.")
        if len(sources) != len(targets):
            raise ValueError("sources and targets must have the same length")
        self.sources = [list(ids) for ids in sources]
        self.targets = [list(ids) for ids in targets]
        for src, tgt in zip(self.sources, self.targets):
            if not src or not tgt:
                raise ValueError("every sequence must contain at least one token id")
            if any(int(token) < UNK_ID for token in src):
                raise ValueError("source content tokens must be >= UNK_ID")
            if any(int(token) < UNK_ID for token in tgt):
                raise ValueError("target content tokens must be >= UNK_ID")

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        src = self.sources[index]
        tgt = self.targets[index]
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
class ProcessedParallel:
    dataset: ParallelDataset
    src_vocab: Vocab
    tgt_vocab: Vocab
    src_text: list[list[str]]
    tgt_text: list[list[str]]

    @property
    def src_vocab_size(self) -> int:
        return self.src_vocab.size

    @property
    def tgt_vocab_size(self) -> int:
        return self.tgt_vocab.size

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


def _encode_pairs(
    pairs: list[tuple[list[str], list[str]]],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    *,
    reverse_source: bool,
) -> tuple[list[list[int]], list[list[int]], list[list[str]], list[list[str]]]:
    sources: list[list[int]] = []
    targets: list[list[int]] = []
    src_text: list[list[str]] = []
    tgt_text: list[list[str]] = []
    for src_tokens, tgt_tokens in pairs:
        src_text.append(list(src_tokens))
        tgt_text.append(list(tgt_tokens))
        sources.append(src_vocab.encode(maybe_reverse(src_tokens, reverse_source)))
        targets.append(tgt_vocab.encode(tgt_tokens))
    return sources, targets, src_text, tgt_text


def process_pairs(
    pairs: list[tuple[list[str], list[str]]],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    config: LuongConfig,
    *,
    filter_length: bool = True,
    max_examples: int | None = None,
) -> ProcessedParallel:
    config.validate()
    if filter_length:
        pairs = filter_by_length(pairs, config.max_len)
    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be >= 1")
        pairs = pairs[:max_examples]
    if not pairs:
        raise ValueError("no parallel pairs left after filtering")
    sources, targets, src_text, tgt_text = _encode_pairs(
        pairs,
        src_vocab,
        tgt_vocab,
        reverse_source=config.reverse_source,
    )
    return ProcessedParallel(
        dataset=ParallelDataset(sources, targets),
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_text=src_text,
        tgt_text=tgt_text,
    )


def process_parallel(
    src_path: Path,
    tgt_path: Path,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    config: LuongConfig,
    *,
    filter_length: bool = True,
    max_examples: int | None = None,
) -> ProcessedParallel:
    pairs = load_parallel_lines(src_path, tgt_path)
    return process_pairs(
        pairs,
        src_vocab,
        tgt_vocab,
        config,
        filter_length=filter_length,
        max_examples=max_examples,
    )


def smoke_parallel(config: LuongConfig | None = None) -> ProcessedParallel:
    config = LuongConfig.smoke() if config is None else config
    pairs = smoke_token_pairs()
    src_vocab = build_vocab(token for src, _ in pairs for token in src)
    tgt_vocab = build_vocab(token for _, tgt in pairs for token in tgt)
    return process_pairs(pairs, src_vocab, tgt_vocab, config, filter_length=False)


def trim_prediction(
    seq: torch.Tensor,
    *,
    pad_id: int = PAD_ID,
    eos_id: int = EOS_ID,
) -> list[int]:
    """Tokens up to and including the first EOS, ignoring trailing PAD."""
    out: list[int] = []
    for token in seq.tolist():
        if token == pad_id:
            break
        out.append(int(token))
        if token == eos_id:
            break
    return out
