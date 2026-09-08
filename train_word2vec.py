"""Train skip-gram or CBOW with hierarchical softmax or negative sampling."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
from torch import nn

from word2vec.config import ProcessingConfig
from word2vec.hierarchical_softmax import HierarchicalSoftmax
from word2vec.negative_sampling import NegativeSampling
from word2vec.pipeline import ProcessedCorpus, process_corpus

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "data" / "text8m1.txt"
EMBEDDING_DIM = 24
BATCH_SIZE = 256
LEARNING_RATE = 0.025
EPOCHS = 200


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
    )
    return process_corpus(path, config)


def _device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    return device


def _log_corpus(processed: ProcessedCorpus) -> None:
    print(f"corpus: {DEFAULT_CORPUS}")
    print(f"raw tokens: {processed.raw_token_count:,}")
    print(f"encoded tokens (before per-epoch subsample): {len(processed.token_ids):,}")
    print(f"vocab size: {len(processed.vocab):,}")
    print(f"architecture: {processed.config.architecture}")
    print("windows are rebuilt each epoch after a new subsample draw")


def _fit(
    model: nn.Module,
    processed: ProcessedCorpus,
    device: torch.device,
    batch_loss: Callable[[nn.Module, dict[str, torch.Tensor], torch.device], torch.Tensor],
    with_negatives: bool,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        loader = processed.dataloader(
            BATCH_SIZE,
            epoch=epoch,
            shuffle=True,
            with_negatives=with_negatives,
        )
        epoch_loss = 0.0
        epoch_pairs = 0
        for batch in loader:
            loss = batch_loss(model, batch, device)
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


def _input_and_target(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    architecture: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    center = batch["center"].to(device)
    context = batch["context"].to(device)
    if architecture == "cbow":
        return context, center
    return center, context


def _hs_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    architecture: str,
) -> torch.Tensor:
    inputs, target = _input_and_target(batch, device, architecture)
    return model(inputs, target)


def _neg_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    architecture: str,
) -> torch.Tensor:
    inputs, target = _input_and_target(batch, device, architecture)
    negatives = batch["negatives"].to(device)
    return model(inputs, target, negatives)


def hierarchical_softmax(architecture: str = "skipgram") -> None:
    processed = process(
        build_huffman=True,
        build_negative_sampler=False,
        architecture=architecture,
    )
    if processed.vocab.coding is None:
        raise RuntimeError("Huffman codes are required for hierarchical softmax.")
    _log_corpus(processed)
    device = _device()
    model = HierarchicalSoftmax(
        processed.vocab.coding,
        embedding_dim=EMBEDDING_DIM,
        vocab_size=len(processed.vocab),
    ).to(device)
    _fit(
        model,
        processed,
        device,
        partial(_hs_loss, architecture=architecture),
        with_negatives=False,
    )


def negative_sampling(architecture: str = "skipgram") -> None:
    processed = process(
        build_huffman=False,
        build_negative_sampler=True,
        architecture=architecture,
    )
    _log_corpus(processed)
    device = _device()
    model = NegativeSampling(
        embedding_dim=EMBEDDING_DIM,
        vocab_size=len(processed.vocab),
    ).to(device)
    _fit(
        model,
        processed,
        device,
        partial(_neg_loss, architecture=architecture),
        with_negatives=True,
    )


if __name__ == "__main__":
    negative_sampling()
    # hierarchical_softmax()
