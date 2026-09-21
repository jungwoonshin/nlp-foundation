"""Reproduce Kim (2014) CNN-rand on TREC question classification."""

from __future__ import annotations

import argparse
import random

import torch

from scripts.textcnn.prepare import prepare_trec
from textcnn.config import TextCNNConfig
from textcnn.dataset import SentenceDataset
from textcnn.model import TextCNN
from textcnn.pipeline import ProcessedSentences, encode_sentences, process_sentences
from textcnn.training import ROOT, device, fit, log_corpus

DATA_DIR = ROOT / "data"
PAPER_ACCURACY = 0.912
SEED = 42


def _holdout(
    processed: ProcessedSentences, fraction: float, seed: int
) -> tuple[ProcessedSentences, ProcessedSentences]:
    n = len(processed.dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(n * fraction)))
    val_ids = set(indices[:n_val])
    train_x: list[list[int]] = []
    train_y: list[int] = []
    val_x: list[list[int]] = []
    val_y: list[int] = []
    for index in range(n):
        item = processed.dataset[index]
        tokens = item["tokens"].tolist()
        label = int(item["label"])
        if index in val_ids:
            val_x.append(tokens)
            val_y.append(label)
        else:
            train_x.append(tokens)
            train_y.append(label)
    train = ProcessedSentences(
        vocab=processed.vocab,
        dataset=SentenceDataset(train_x, train_y),
        label_to_id=processed.label_to_id,
        config=processed.config,
    )
    val = ProcessedSentences(
        vocab=processed.vocab,
        dataset=SentenceDataset(val_x, val_y),
        label_to_id=processed.label_to_id,
        config=processed.config,
    )
    return train, val


def run(*, paper: bool = False) -> dict[str, float]:
    torch.manual_seed(SEED)
    random.seed(SEED)
    train_path, test_path = prepare_trec(DATA_DIR)
    if paper:
        config = TextCNNConfig(
            embedding_dim=300,
            min_count=1,
            dropout=0.5,
            filter_sizes=(3, 4, 5),
            num_filters=100,
            max_length=100,
            seed=SEED,
        )
        epochs = 25
        batch_size = 50
        learning_rate = 1.0
        optimizer_name = "adadelta"
        max_norm = 3.0
        val_fraction = 0.1
    else:
        config = TextCNNConfig()
        epochs = 10
        batch_size = 64
        learning_rate = 1e-3
        optimizer_name = "adam"
        max_norm = None
        val_fraction = None

    train_all = process_sentences(train_path, config)
    test = encode_sentences(test_path, train_all)
    if val_fraction is not None:
        train, val = _holdout(train_all, val_fraction, SEED)
    else:
        train, val = train_all, None

    print("split: TREC train/test")
    log_corpus(train, train_path)
    print(f"test documents: {len(test.dataset):,}")
    if val is not None:
        print(f"val documents: {len(val.dataset):,}")
    print(f"optimizer: {optimizer_name}  epochs: {epochs}  batch_size: {batch_size}")
    print(f"paper CNN-rand accuracy: {PAPER_ACCURACY:.3f}")

    chosen = device()
    model = TextCNN(
        vocab_size=train.vocab_size,
        embedding_dim=config.embedding_dim,
        num_classes=len(train.label_to_id),
        filter_sizes=config.filter_sizes,
        num_filters=config.num_filters,
        dropout=config.dropout,
    ).to(chosen)
    metrics = fit(
        model,
        train,
        chosen,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        eval_processed=test,
        val_processed=val,
        max_norm=max_norm,
    )
    selected = metrics["best_test_acc"] if val is not None else metrics["test_acc"]
    print(
        f"selected test accuracy: {selected:.4f}  "
        f"paper CNN-rand: {PAPER_ACCURACY:.4f}  "
        f"delta: {selected - PAPER_ACCURACY:+.4f}"
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Use Kim (2014) CNN-rand training settings instead of repo defaults.",
    )
    args = parser.parse_args()
    run(paper=args.paper)
