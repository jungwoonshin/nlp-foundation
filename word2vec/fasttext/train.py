"""Train supervised FastText on labeled documents."""

from __future__ import annotations

from pathlib import Path

from prepare_ag_news import prepare_ag_news
from word2vec.fasttext.model import BOW_FastText
from word2vec.training import (
    EMBEDDING_DIM,
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


def fasttext(*, smoke: bool = False) -> None:
    tmp = tiny_corpus(labeled=True) if smoke else None
    try:
        if smoke:
            assert tmp is not None
            processed, path = smoke_process(tmp, architecture="fasttext")
        else:
            path = prepare_ag_news(DATA_DIR)
            processed = process(
                path,
                build_huffman=False,
                build_negative_sampler=False,
                architecture="fasttext",
                max_sentences=100,
            )
        if processed.label_to_id is None:
            raise RuntimeError("FastText needs class labels on every document.")
        log_corpus(processed, Path(path))
        chosen = device()
        model = BOW_FastText(
            embedding_dim=EMBEDDING_DIM,
            vocab=processed.vocab,
            num_classes=len(processed.label_to_id),
        ).to(chosen)
        fit(
            model,
            processed,
            chosen,
            classification_loss,
            with_negatives=False,
            epochs=1 if smoke else None,
            batch_size=2 if smoke else None,
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    fasttext(smoke=True)
