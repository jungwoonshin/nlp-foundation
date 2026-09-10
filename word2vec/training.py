"""Shared training helpers for the four paper implementations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from word2vec.config import ProcessingConfig
from word2vec.pipeline import ProcessedCorpus, process_corpus

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "data" / "text8m1.txt"
EMBEDDING_DIM = 24
BATCH_SIZE = 256
LEARNING_RATE = 0.025
EPOCHS = 200
TINY_CORPUS_TEXT = "alpha beta gamma delta epsilon zeta eta theta\n"


def process(
    path: str | Path = DEFAULT_CORPUS,
    *,
    min_count: int = 5,
    window_size: int = 5,
    subsample_threshold: float = 1e-3,
    num_negatives: int = 5,
    seed: int = 42,
    build_huffman: bool = False,
    build_negative_sampler: bool = False,
    architecture: str = "skipgram",
    max_sentences: int | None = None,
    max_examples: int | None = None,
    negative_table_size: int = 1_000_000,
) -> ProcessedCorpus:
    """Load `path` and return a PyTorch Dataset of skip-gram or CBOW examples."""

    config = ProcessingConfig(
        min_count=min_count,
        window_size=window_size,
        subsample_threshold=subsample_threshold,
        num_negatives=num_negatives,
        seed=seed,
        build_huffman=build_huffman,
        build_negative_sampler=build_negative_sampler,
        architecture=architecture,
        max_sentences=max_sentences,
        max_examples=max_examples,
        negative_table_size=negative_table_size,
    )
    return process_corpus(path, config)


def device() -> torch.device:
    chosen = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {chosen}")
    return chosen


def log_corpus(processed: ProcessedCorpus, path: Path | None = None) -> None:
    print(f"corpus: {path or DEFAULT_CORPUS}")
    print(f"raw tokens: {processed.raw_token_count:,}")
    print(f"encoded tokens (before per-epoch subsample): {sum(len(s) for s in processed.sentences):,}")
    print(f"vocab size: {len(processed.vocab):,}")
    print(f"architecture: {processed.config.architecture}")
    print("each epoch: subsample each sentence, cut at max_sentence_length, then windows")


def fit(
    model: nn.Module,
    processed: ProcessedCorpus,
    chosen_device: torch.device,
    batch_loss: Callable[[nn.Module, dict[str, torch.Tensor], torch.device], torch.Tensor],
    with_negatives: bool,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> None:
    epochs = EPOCHS if epochs is None else epochs
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    learning_rate = LEARNING_RATE if learning_rate is None else learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(1, epochs + 1):
        loader = processed.dataloader(
            batch_size,
            epoch=epoch,
            shuffle=True,
            with_negatives=with_negatives,
        )
        epoch_loss = 0.0
        epoch_pairs = 0
        for batch in loader:
            loss = batch_loss(model, batch, chosen_device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_pairs = int(batch["center"].shape[0])
            epoch_loss += float(loss) * batch_pairs
            epoch_pairs += batch_pairs
        print(
            f"epoch {epoch:3d}  kept={processed.kept_token_count:,}  "
            f"examples={epoch_pairs:,}  loss={epoch_loss / max(epoch_pairs, 1):.4f}"
        )


def input_and_target(
    batch: dict[str, torch.Tensor],
    chosen_device: torch.device,
    architecture: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    center = batch["center"].to(chosen_device)
    context = batch["context"].to(chosen_device)
    if architecture == "cbow":
        return context, center
    return center, context


def hs_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    chosen_device: torch.device,
    architecture: str,
) -> torch.Tensor:
    inputs, target = input_and_target(batch, chosen_device, architecture)
    return model(inputs, target)


def neg_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    chosen_device: torch.device,
    architecture: str,
) -> torch.Tensor:
    inputs, target = input_and_target(batch, chosen_device, architecture)
    negatives = batch["negatives"].to(chosen_device)
    return model(inputs, target, negatives)


def write_tiny_corpus(path: Path) -> Path:
    path.write_text(TINY_CORPUS_TEXT, encoding="utf-8")
    return path


def tiny_corpus() -> TemporaryDirectory[str]:
    tmp = TemporaryDirectory()
    write_tiny_corpus(Path(tmp.name) / "tiny.txt")
    return tmp


def smoke_process(
    tmp: TemporaryDirectory[str],
    *,
    build_huffman: bool = False,
    build_negative_sampler: bool = False,
    architecture: str = "skipgram",
) -> tuple[ProcessedCorpus, Path]:
    path = Path(tmp.name) / "tiny.txt"
    processed = process(
        path,
        min_count=1,
        window_size=1,
        subsample_threshold=1.0,
        num_negatives=1,
        seed=0,
        build_huffman=build_huffman,
        build_negative_sampler=build_negative_sampler,
        architecture=architecture,
        max_examples=2,
        negative_table_size=64,
    )
    return processed, path
