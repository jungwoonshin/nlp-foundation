"""Train supervised FastText on labeled documents."""

from __future__ import annotations

from pathlib import Path

from prepare_ag_news import prepare_ag_news
from word2vec.fasttext.model import BOW_FastText
from word2vec.training import (
    ROOT,
    classification_loss,
    device,
    fit,
    log_corpus,
    process,
    smoke_process,
    tiny_corpus,
)

DATA_DIR = ROOT / "data"

# Joulin et al. 2017 Table 1 / official classification-results.sh (AG News).
PAPER_EMBEDDING_DIM = 10
PAPER_EPOCHS = 5
PAPER_LEARNING_RATE = 0.25
PAPER_MIN_COUNT = 1
PAPER_NUM_BUCKETS = 10_000_000
PAPER_NGRAM_SIZE = 2
PAPER_ACCURACY = 0.925
# Official FastText is online SGD; this is a practical mini-batch stand-in.
PAPER_BATCH_SIZE = 32


def _sum_classification_loss(model, batch, chosen_device):
    """Undo batch-mean so SGD lr=0.25 matches per-example FastText updates."""
    mean_loss = classification_loss(model, batch, chosen_device)
    return mean_loss * int(batch["label"].shape[0])


def fasttext(*, smoke: bool = False) -> None:
    tmp = tiny_corpus(labeled=True) if smoke else None
    try:
        if smoke:
            assert tmp is not None
            processed, path = smoke_process(tmp, architecture="fasttext")
            embedding_dim = 8
            num_buckets = 64
            epochs = 1
            batch_size = 2
            learning_rate = 0.25
            min_count = 1
        else:
            path, _ = prepare_ag_news(DATA_DIR)
            embedding_dim = PAPER_EMBEDDING_DIM
            num_buckets = PAPER_NUM_BUCKETS
            epochs = PAPER_EPOCHS
            batch_size = PAPER_BATCH_SIZE
            learning_rate = PAPER_LEARNING_RATE
            min_count = PAPER_MIN_COUNT
            processed = process(
                path,
                min_count=min_count,
                build_huffman=False,
                build_negative_sampler=False,
                architecture="fasttext",
                ngram_size=PAPER_NGRAM_SIZE,
                num_buckets=num_buckets,
            )
        if processed.label_to_id is None:
            raise RuntimeError("FastText needs class labels on every document.")
        log_corpus(processed, Path(path))
        chosen = device()
        model = BOW_FastText(
            embedding_dim=embedding_dim,
            vocab=processed.vocab,
            num_classes=len(processed.label_to_id),
            num_buckets=num_buckets,
        ).to(chosen)
        fit(
            model,
            processed,
            chosen,
            _sum_classification_loss,
            with_negatives=False,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_name="sgd",
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    fasttext(smoke=True)
