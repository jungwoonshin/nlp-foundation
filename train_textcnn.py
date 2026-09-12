"""Train Yoon Kim CNN-rand on labeled documents."""

from __future__ import annotations

from pathlib import Path

from prepare_ag_news import prepare_ag_news
from textcnn.config import TextCNNConfig
from textcnn.model import TextCNN
from textcnn.pipeline import process_sentences
from textcnn.training import (
    ROOT,
    device,
    fit,
    log_corpus,
    smoke_process,
    tiny_labeled_corpus,
)

DATA_DIR = ROOT / "data"


def textcnn(*, smoke: bool = False) -> None:
    tmp = tiny_labeled_corpus() if smoke else None
    try:
        if smoke:
            assert tmp is not None
            processed, path = smoke_process(tmp)
        else:
            path = prepare_ag_news(DATA_DIR)
            processed = process_sentences(
                path,
                TextCNNConfig(max_sentences=100),
            )
        log_corpus(processed, Path(path))
        chosen = device()
        model = TextCNN(
            vocab_size=processed.vocab_size,
            embedding_dim=processed.config.embedding_dim,
            num_classes=len(processed.label_to_id),
            filter_sizes=processed.config.filter_sizes,
            num_filters=processed.config.num_filters,
            dropout=processed.config.dropout,
        ).to(chosen)
        fit(
            model,
            processed,
            chosen,
            epochs=1 if smoke else None,
            batch_size=2 if smoke else None,
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    textcnn(smoke=True)
