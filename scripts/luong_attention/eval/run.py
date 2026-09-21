"""Evaluate Luong attention NMT (BLEU / perplexity) on IWSLT'15 tst2013."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.luong_attention.prepare import prepare_iwslt15
from seq_to_seq.attention.config import LuongConfig
from seq_to_seq.attention.data.dataset import process_parallel, smoke_parallel
from seq_to_seq.attention.data.vocab import load_stanford_vocab
from seq_to_seq.attention.eval.metrics import (
    evaluate_bleu,
    evaluate_exact_match,
    evaluate_perplexity,
)
from seq_to_seq.attention.model.nmt import LuongNMT
from seq_to_seq.attention.training.dummy import DummyNMT
from seq_to_seq.attention.training.loop import ROOT, device, initialize_parameters, log_corpus

DATA_DIR = ROOT / "data"


def _print_metrics(metrics: dict[str, float]) -> None:
    print(
        f"ppl={metrics['perplexity']:.4f}  "
        f"bleu={metrics['bleu']:.2f}  exact_match={metrics['exact_match']:.4f}"
    )


def run(*, smoke: bool = False, checkpoint: Path | None = None) -> dict[str, float] | None:
    chosen = device()
    if smoke:
        config = LuongConfig.smoke()
        processed = smoke_parallel(config)
        log_corpus(processed)
        model = DummyNMT(
            processed.src_vocab_size,
            processed.tgt_vocab_size,
            embed_dim=config.embed_dim,
            hidden_size=config.hidden_size,
        ).to(chosen)
        initialize_parameters(model, config.init_range)
        metrics = {
            "perplexity": evaluate_perplexity(model, processed, chosen, batch_size=config.batch_size),
            "bleu": evaluate_bleu(
                model, processed, chosen, batch_size=config.batch_size, max_len=config.max_len
            ),
            "exact_match": evaluate_exact_match(
                model, processed, chosen, batch_size=config.batch_size
            ),
        }
        _print_metrics(metrics)
        return metrics

    config = LuongConfig()
    config.validate()
    raw = prepare_iwslt15(DATA_DIR)
    src_vocab = load_stanford_vocab(raw / f"vocab.{config.src_lang}")
    tgt_vocab = load_stanford_vocab(raw / f"vocab.{config.tgt_lang}")
    test = process_parallel(
        raw / f"tst2013.{config.src_lang}",
        raw / f"tst2013.{config.tgt_lang}",
        src_vocab,
        tgt_vocab,
        config,
        filter_length=False,
        max_examples=config.max_eval_examples,
    )
    log_corpus(test)
    try:
        model = LuongNMT(test.src_vocab_size, test.tgt_vocab_size, config).to(chosen)
    except NotImplementedError:
        print(
            "LuongNMT is a stub (signatures only). Implement seq_to_seq/attention/model "
            "then train, or pass --smoke to score DummyNMT on the built-in pairs."
        )
        return None
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=chosen)
        model.load_state_dict(state)
        print(f"loaded checkpoint: {checkpoint}")
    metrics = {
        "perplexity": evaluate_perplexity(model, test, chosen, batch_size=config.batch_size),
        "bleu": evaluate_bleu(
            model, test, chosen, batch_size=config.batch_size, max_len=config.max_len
        ),
        "exact_match": evaluate_exact_match(model, test, chosen, batch_size=config.batch_size),
    }
    _print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Score DummyNMT on the tiny smoke pairs.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional state_dict path for a filled LuongNMT.",
    )
    args = parser.parse_args()
    run(smoke=args.smoke, checkpoint=args.checkpoint)
