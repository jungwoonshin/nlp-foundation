"""Reproduce Joulin et al. 2017 supervised FastText on AG News.

Compares this repo's bag-of-tricks classifier to Table 1:
fastText, h=10, bigram, AG News test accuracy = 92.5.
"""

from __future__ import annotations

import argparse

import torch

from prepare_ag_news import prepare_ag_news
from word2vec.config import ProcessingConfig
from word2vec.fasttext.model import BOW_FastText
from word2vec.fasttext.train import (
    PAPER_ACCURACY,
    PAPER_BATCH_SIZE,
    PAPER_EMBEDDING_DIM,
    PAPER_EPOCHS,
    PAPER_LEARNING_RATE,
    PAPER_MIN_COUNT,
    PAPER_NGRAM_SIZE,
    PAPER_NUM_BUCKETS,
)
from word2vec.pipeline import encode_labeled_corpus, process_corpus
from word2vec.training import ROOT, classification_loss, device, fit, log_corpus

DATA_DIR = ROOT / "data"
SEED = 42


def _sum_classification_loss(model, batch, chosen_device):
    """Undo batch-mean so SGD lr=0.25 matches per-example FastText updates."""
    mean_loss = classification_loss(model, batch, chosen_device)
    return mean_loss * int(batch["label"].shape[0])


@torch.no_grad()
def accuracy(
    model: BOW_FastText,
    processed,
    chosen: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    loader = processed.dataloader(
        batch_size, epoch=1, shuffle=False, with_negatives=False
    )
    correct = 0
    total = 0
    for batch in loader:
        hidden = model.encode(
            batch["features"].to(chosen),
            batch["weights"].to(chosen),
            batch["ngrams"].to(chosen),
        )
        pred = model.classifier(hidden).argmax(dim=-1)
        labels = batch["label"].to(chosen)
        correct += int((pred == labels).sum())
        total += int(labels.shape[0])
    model.train()
    return correct / max(total, 1)


def run() -> dict[str, float]:
    torch.manual_seed(SEED)
    train_path, test_path = prepare_ag_news(DATA_DIR)
    config = ProcessingConfig(
        min_count=PAPER_MIN_COUNT,
        architecture="fasttext",
        ngram_size=PAPER_NGRAM_SIZE,
        num_buckets=PAPER_NUM_BUCKETS,
        seed=SEED,
    )
    train = process_corpus(train_path, config)
    test = encode_labeled_corpus(test_path, train)
    print("split: AG News train/test")
    log_corpus(train, train_path)
    print(f"test documents: {len(test.dataset):,}")
    print(
        f"optimizer: sgd  epochs: {PAPER_EPOCHS}  batch_size: {PAPER_BATCH_SIZE}  "
        f"lr: {PAPER_LEARNING_RATE}  dim: {PAPER_EMBEDDING_DIM}"
    )
    print(f"paper fastText h=10 bigram accuracy: {PAPER_ACCURACY:.3f}")

    chosen = device()
    if train.label_to_id is None:
        raise RuntimeError("FastText needs class labels on every document.")
    model = BOW_FastText(
        embedding_dim=PAPER_EMBEDDING_DIM,
        vocab=train.vocab,
        num_classes=len(train.label_to_id),
        num_buckets=PAPER_NUM_BUCKETS,
    ).to(chosen)

    history: list[float] = []

    def after_epoch(epoch: int, trained: torch.nn.Module, mean_loss: float) -> None:
        acc = accuracy(trained, test, chosen, PAPER_BATCH_SIZE)
        history.append(acc)
        print(
            f"epoch {epoch:3d}  test_acc={acc:.4f}  "
            f"paper={PAPER_ACCURACY:.4f}  delta={acc - PAPER_ACCURACY:+.4f}"
        )

    fit(
        model,
        train,
        chosen,
        _sum_classification_loss,
        with_negatives=False,
        epochs=PAPER_EPOCHS,
        batch_size=PAPER_BATCH_SIZE,
        learning_rate=PAPER_LEARNING_RATE,
        optimizer_name="sgd",
        log_every=500,
        after_epoch=after_epoch,
    )
    final_acc = history[-1] if history else accuracy(model, test, chosen, PAPER_BATCH_SIZE)
    print(
        f"selected test accuracy: {final_acc:.4f}  "
        f"paper fastText h=10 bigram: {PAPER_ACCURACY:.4f}  "
        f"delta: {final_acc - PAPER_ACCURACY:+.4f}"
    )
    return {"test_acc": final_acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
