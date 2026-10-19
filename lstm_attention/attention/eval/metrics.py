from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from lstm_attention.attention.data.dataset import ProcessedParallel, trim_prediction
from lstm_attention.attention.data.vocab import PAD_ID, Vocab
from lstm_attention.attention.eval.bleu import corpus_bleu

BATCH_SIZE = 32


def ids_to_tokens(seq: torch.Tensor, vocab: Vocab) -> list[str]:
    return vocab.decode(trim_prediction(seq), stop_at_eos=True)


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    processed: ProcessedParallel,
    chosen_device: torch.device,
    *,
    batch_size: int | None = None,
) -> float:
    """Teacher-forced exp(mean NLL), ignoring PAD. Model is left in eval mode."""
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in processed.dataloader(batch_size, shuffle=False):
        src = batch["src"].to(chosen_device)
        lengths = batch["src_lengths"].to(chosen_device)
        tgt_in = batch["tgt_in"].to(chosen_device)
        tgt_out = batch["tgt_out"].to(chosen_device)
        logits = model(src, lengths, tgt_in)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=PAD_ID,
            reduction="sum",
        )
        total_nll += float(nll)
        total_tokens += int((tgt_out != PAD_ID).sum().item())
    mean_nll = total_nll / max(total_tokens, 1)
    return math.exp(mean_nll)


@torch.no_grad()
def evaluate_bleu(
    model: nn.Module,
    processed: ProcessedParallel,
    chosen_device: torch.device,
    *,
    batch_size: int | None = None,
    max_len: int | None = None,
) -> float:
    """Greedy corpus BLEU against stored target token lists."""
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    model.eval()
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise TypeError("BLEU evaluation requires a model with a callable generate() method")
    hypotheses: list[list[str]] = []
    references: list[list[str]] = []
    offset = 0
    for batch in processed.dataloader(batch_size, shuffle=False):
        src = batch["src"].to(chosen_device)
        lengths = batch["src_lengths"].to(chosen_device)
        tgt_out = batch["tgt_out"]
        decode_len = int(tgt_out.shape[1]) if max_len is None else max_len
        pred = generate(src, lengths, max_len=decode_len)
        if not isinstance(pred, torch.Tensor):
            raise TypeError("generate() must return a tensor of token IDs")
        n = int(src.shape[0])
        for row in range(n):
            hypotheses.append(ids_to_tokens(pred[row], processed.tgt_vocab))
            references.append(list(processed.tgt_text[offset + row]))
        offset += n
    return corpus_bleu(hypotheses, references)


@torch.no_grad()
def evaluate_exact_match(
    model: nn.Module,
    processed: ProcessedParallel,
    chosen_device: torch.device,
    *,
    batch_size: int | None = None,
) -> float:
    """Fraction of sequences whose greedy ids match tgt_out up to EOS."""
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    model.eval()
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise TypeError("Exact-match evaluation requires a model with a callable generate() method")
    correct = 0
    total = 0
    for batch in processed.dataloader(batch_size, shuffle=False):
        src = batch["src"].to(chosen_device)
        lengths = batch["src_lengths"].to(chosen_device)
        tgt_out = batch["tgt_out"].to(chosen_device)
        pred = generate(src, lengths, max_len=int(tgt_out.shape[1]))
        if not isinstance(pred, torch.Tensor):
            raise TypeError("generate() must return a tensor of token IDs")
        for row in range(src.shape[0]):
            if trim_prediction(pred[row]) == trim_prediction(tgt_out[row]):
                correct += 1
        total += int(src.shape[0])
    return correct / max(total, 1)
