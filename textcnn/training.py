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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    processed: ProcessedSentences,
    chosen_device: torch.device,
    *,
    batch_size: int | None = None,
) -> float:
    """Accuracy on a labeled split. Model is left in eval mode."""
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    model.eval()
    correct = 0
    total = 0
    for batch in processed.dataloader(batch_size, shuffle=False):
        tokens = batch["tokens"].to(chosen_device)
        labels = batch["label"].to(chosen_device)
        pred = model(tokens).argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.shape[0])
    return correct / max(total, 1)


def constrain_l2_rows(weight: torch.Tensor, max_norm: float) -> None:
    """Kim (2014): rescale each row of a 2-D weight if its L2 norm exceeds s."""
    with torch.no_grad():
        norms = weight.norm(p=2, dim=1, keepdim=True)
        scale = norms.clamp(max=max_norm) / (1e-7 + norms)
        weight.mul_(scale)


def fit(
    model: nn.Module,
    processed: ProcessedSentences,
    chosen_device: torch.device,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    optimizer_name: str = "adam",
    eval_processed: ProcessedSentences | None = None,
    val_processed: ProcessedSentences | None = None,
    max_norm: float | None = None,
) -> dict[str, float]:
    epochs = EPOCHS if epochs is None else epochs
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    learning_rate = LEARNING_RATE if learning_rate is None else learning_rate
    name = optimizer_name.lower()
    if name == "adadelta":
        rho = 0.95
        optimizer = torch.optim.Adadelta(
            model.parameters(), lr=learning_rate, rho=rho, eps=1e-6
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"unknown optimizer: {optimizer_name}")

    best_val = -1.0
    best_test = 0.0
    last_test = 0.0
    last_val = 0.0
    last_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
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
            if max_norm is not None and hasattr(model, "fc"):
                constrain_l2_rows(model.fc.weight, max_norm)
            n = int(labels.shape[0])
            epoch_loss += float(loss) * n
            epoch_docs += n
        last_loss = epoch_loss / max(epoch_docs, 1)
        parts = [
            f"epoch {epoch:3d}  documents={epoch_docs:,}  ",
            f"loss={last_loss:.4f}",
        ]
        if val_processed is not None:
            last_val = evaluate(
                model, val_processed, chosen_device, batch_size=batch_size
            )
            parts.append(f"  val_acc={last_val:.4f}")
        if eval_processed is not None:
            last_test = evaluate(
                model, eval_processed, chosen_device, batch_size=batch_size
            )
            parts.append(f"  test_acc={last_test:.4f}")
            if val_processed is None or last_val >= best_val:
                if val_processed is not None:
                    best_val = last_val
                best_test = last_test
        print("".join(parts))
    return {
        "loss": last_loss,
        "val_acc": last_val,
        "test_acc": last_test,
        "best_test_acc": best_test if eval_processed is not None else last_test,
        "best_val_acc": best_val if val_processed is not None else last_val,
    }


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
