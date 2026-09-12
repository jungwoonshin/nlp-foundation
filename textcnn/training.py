"""Training loop for Yoon Kim CNN-rand. Loss is computed here, not in the model."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn import functional as F

from textcnn.config import TextCNNConfig
from textcnn.pipeline import ProcessedSentences, process_sentences

ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
TINY_LABELED_CORPUS_TEXT = (
    "__label__3 alpha beta alpha gamma\n__label__2 delta epsilon delta zeta\n"
)


def device() -> torch.device:
    chosen = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {chosen}")
    return chosen


def log_corpus(processed: ProcessedSentences, path: Path | None = None) -> None:
    if path is not None:
        print(f"corpus: {path}")
    print(f"documents: {len(processed.dataset):,}")
    print(f"vocab size (words): {len(processed.vocab):,}")
    print(f"embedding rows (pad + words): {processed.vocab_size:,}")
    print(f"classes: {len(processed.label_to_id)}")
    print(f"filter_sizes: {processed.config.filter_sizes}")


def fit(
    model: nn.Module,
    processed: ProcessedSentences,
    chosen_device: torch.device,
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
        loader = processed.dataloader(batch_size, shuffle=True)
        epoch_loss = 0.0
        epoch_docs = 0
        for batch in loader:
            tokens = batch["tokens"].to(chosen_device)
            labels = batch["label"].to(chosen_device)
            logits = model(tokens)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n = int(labels.shape[0])
            epoch_loss += float(loss) * n
            epoch_docs += n
        print(
            f"epoch {epoch:3d}  documents={epoch_docs:,}  "
            f"loss={epoch_loss / max(epoch_docs, 1):.4f}"
        )


def write_tiny_corpus(path: Path, text: str = TINY_LABELED_CORPUS_TEXT) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def tiny_labeled_corpus() -> TemporaryDirectory[str]:
    tmp = TemporaryDirectory()
    write_tiny_corpus(Path(tmp.name) / "tiny.txt")
    return tmp


def smoke_process(tmp: TemporaryDirectory[str]) -> tuple[ProcessedSentences, Path]:
    path = Path(tmp.name) / "tiny.txt"
    processed = process_sentences(
        path,
        TextCNNConfig(
            min_count=1,
            embedding_dim=8,
            num_filters=4,
            max_length=16,
            max_examples=2,
        ),
    )
    return processed, path
