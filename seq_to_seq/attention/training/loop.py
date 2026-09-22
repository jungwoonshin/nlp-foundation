"""Training loop for Luong attention NMT. Loss is computed here, not in the model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from seq_to_seq.attention.config import LuongConfig
from seq_to_seq.attention.data.dataset import ProcessedParallel
from seq_to_seq.attention.data.vocab import PAD_ID
from seq_to_seq.attention.eval.metrics import (
    evaluate_bleu,
    evaluate_exact_match,
    evaluate_perplexity,
)

ROOT = Path(__file__).resolve().parent.parent.parent.parent


def device() -> torch.device:
    chosen = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {chosen}")
    return chosen


def log_corpus(processed: ProcessedParallel) -> None:
    print(f"examples: {len(processed.dataset):,}")
    print(f"src vocab size: {processed.src_vocab_size:,}")
    print(f"tgt vocab size: {processed.tgt_vocab_size:,}")


def initialize_parameters(model: nn.Module, init_range: float = 0.1) -> None:
    """Paper: uniform in [-0.1, 0.1]. No-op when the module has no parameters."""
    if init_range <= 0.0:
        raise ValueError("init_range must be > 0")
    for param in model.parameters():
        nn.init.uniform_(param, -init_range, init_range)


def learning_rate_for_epoch(epoch: int, config: LuongConfig) -> float:
    """SGD: keep lr until lr_decay_start, then halve every epoch. Adam: constant."""
    if epoch < 1:
        raise ValueError("epoch must be >= 1")
    if config.optimizer != "sgd":
        return config.learning_rate
    if epoch <= config.lr_decay_start:
        return config.learning_rate
    return config.learning_rate * (0.5 ** (epoch - config.lr_decay_start))


def _make_optimizer(model: nn.Module, config: LuongConfig, learning_rate: float) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    if config.optimizer == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    if config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=learning_rate)
    raise ValueError(f"unknown optimizer: {config.optimizer}")


def fit(
    model: nn.Module,
    processed: ProcessedParallel,
    chosen_device: torch.device,
    config: LuongConfig,
    *,
    eval_processed: ProcessedParallel | None = None,
    eval_bleu: bool = False,
    verbose: bool = True,
) -> dict[str, float]:
    config.validate()
    optimizer = _make_optimizer(model, config, learning_rate_for_epoch(1, config))
    last_loss = 0.0
    last_ppl = 0.0
    for epoch in range(1, config.epochs + 1):
        lr = learning_rate_for_epoch(epoch, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        loader = processed.dataloader(config.batch_size, shuffle=True)
        epoch_loss = 0.0
        epoch_docs = 0
        for batch in loader:
            src = batch["src"].to(chosen_device)
            lengths = batch["src_lengths"].to(chosen_device)
            tgt_in = batch["tgt_in"].to(chosen_device)
            tgt_out = batch["tgt_out"].to(chosen_device)
            logits = model(src, lengths, tgt_in)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
                ignore_index=PAD_ID,
            )
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            n = int(src.shape[0])
            epoch_loss += float(loss) * n
            epoch_docs += n
        last_loss = epoch_loss / max(epoch_docs, 1)
        parts = [
            f"epoch {epoch:3d}  examples={epoch_docs:,}  ",
            f"loss={last_loss:.4f}  lr={lr:.4g}",
        ]
        if eval_processed is not None:
            last_ppl = evaluate_perplexity(
                model, eval_processed, chosen_device, batch_size=config.batch_size
            )
            parts.append(f"  ppl={last_ppl:.4f}")
        if verbose:
            print("".join(parts))

    metrics: dict[str, float] = {"loss": last_loss, "perplexity": last_ppl, "bleu": 0.0, "exact_match": 0.0}
    if eval_processed is not None:
        metrics["perplexity"] = evaluate_perplexity(
            model, eval_processed, chosen_device, batch_size=config.batch_size
        )
        metrics["exact_match"] = evaluate_exact_match(
            model, eval_processed, chosen_device, batch_size=config.batch_size
        )
        if eval_bleu:
            metrics["bleu"] = evaluate_bleu(
                model,
                eval_processed,
                chosen_device,
                batch_size=config.batch_size,
                max_len=config.max_len,
            )
            if verbose:
                print(
                    f"eval  ppl={metrics['perplexity']:.4f}  "
                    f"bleu={metrics['bleu']:.2f}  exact_match={metrics['exact_match']:.4f}"
                )
        elif verbose:
            print(
                f"eval  ppl={metrics['perplexity']:.4f}  "
                f"exact_match={metrics['exact_match']:.4f}"
            )
    return metrics
