"""Training loop for reverse-sequence seq2seq. Loss is computed here, not in the model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from seq_to_seq.foundation.common.dataset import (
    PAD_ID,
    ProcessedReverse,
    make_reverse_dataset,
    smoke_reverse,
    trim_prediction,
)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10
FULL_VOCAB_SIZE = 20
FULL_TRAIN_EXAMPLES = 256
FULL_EVAL_EXAMPLES = 64
FULL_MIN_LEN = 3
FULL_MAX_LEN = 8
FULL_EMBED_DIM = 32
FULL_HIDDEN_SIZE = 64
SMOKE_EMBED_DIM = 8
SMOKE_HIDDEN_SIZE = 16

MakeModel = Callable[..., nn.Module]


def device() -> torch.device:
    chosen = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {chosen}")
    return chosen


def log_corpus(processed: ProcessedReverse) -> None:
    print(f"examples: {len(processed.dataset):,}")
    print(f"vocab size: {processed.vocab_size:,}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    processed: ProcessedReverse,
    chosen_device: torch.device,
    *,
    batch_size: int | None = None,
) -> float:
    """Exact-match accuracy on reversed targets. Model is left in eval mode."""
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    model.eval()
    correct = 0
    total = 0
    for batch in processed.dataloader(batch_size, shuffle=False):
        src = batch["src"].to(chosen_device)
        lengths = batch["src_lengths"].to(chosen_device)
        tgt_out = batch["tgt_out"].to(chosen_device)
        pred = model.generate(src, lengths, max_len=int(tgt_out.shape[1]))
        for row in range(src.shape[0]):
            if trim_prediction(pred[row]) == trim_prediction(tgt_out[row]):
                correct += 1
        total += int(src.shape[0])
    return correct / max(total, 1)


def fit(
    model: nn.Module,
    processed: ProcessedReverse,
    chosen_device: torch.device,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    eval_processed: ProcessedReverse | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    epochs = EPOCHS if epochs is None else epochs
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    learning_rate = LEARNING_RATE if learning_rate is None else learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    last_loss = 0.0
    last_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        loader = processed.dataloader(batch_size, shuffle=True)
        epoch_loss = 0.0
        epoch_docs = 0
        for batch in loader:
            src = batch["src"].to(chosen_device)
            lengths = batch["src_lengths"].to(chosen_device)
            tgt_in = batch["tgt_in"].to(chosen_device) # (batch, seq_len)
            tgt_out = batch["tgt_out"].to(chosen_device) # (batch, seq_len)
            logits = model(src, lengths, tgt_in) # (batch, seq_len, vocab_size)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
                ignore_index=PAD_ID,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n = int(src.shape[0])
            epoch_loss += float(loss) * n
            epoch_docs += n
        last_loss = epoch_loss / max(epoch_docs, 1)
        parts = [
            f"epoch {epoch:3d}  examples={epoch_docs:,}  ",
            f"loss={last_loss:.4f}",
        ]
        if eval_processed is not None:
            last_acc = evaluate(
                model, eval_processed, chosen_device, batch_size=batch_size
            )
            parts.append(f"  exact_match={last_acc:.4f}")
        if verbose:
            print("".join(parts))
    return {"loss": last_loss, "exact_match": last_acc}


def run_reverse(make_model: MakeModel, *, smoke: bool = False) -> dict[str, float]:
    chosen = device()
    if smoke:
        processed = smoke_reverse()
        eval_processed = processed
        embed_dim = SMOKE_EMBED_DIM
        hidden_size = SMOKE_HIDDEN_SIZE
        epochs = 1
        batch_size = 2
    else:
        processed = make_reverse_dataset(
            FULL_TRAIN_EXAMPLES,
            FULL_VOCAB_SIZE,
            min_len=FULL_MIN_LEN,
            max_len=FULL_MAX_LEN,
            seed=0,
        )
        eval_processed = make_reverse_dataset(
            FULL_EVAL_EXAMPLES,
            FULL_VOCAB_SIZE,
            min_len=FULL_MIN_LEN,
            max_len=FULL_MAX_LEN,
            seed=1,
        )
        embed_dim = FULL_EMBED_DIM
        hidden_size = FULL_HIDDEN_SIZE
        epochs = None
        batch_size = None

    log_corpus(processed)
    torch_model = make_model(
        processed.vocab_size,
        backend="torch",
        embed_dim=embed_dim,
        hidden_size=hidden_size,
    )
    print(f"constructed backends: scratch, torch  ({type(torch_model).__module__})")
    model = make_model(
        processed.vocab_size,
        backend="scratch",
        embed_dim=embed_dim,
        hidden_size=hidden_size,
    ).to(chosen)
    return fit(
        model,
        processed,
        chosen,
        epochs=epochs,
        batch_size=batch_size,
        eval_processed=eval_processed,
    )
