"""Train FastText skip-gram NEG (BoW / n-gram inputs)."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from prepare_ag_news import prepare_ag_news
from word2vec.fasttext.model import BOW_FastText
from word2vec.training import (
    EMBEDDING_DIM,
    ROOT,
    device,
    fit,
    log_corpus,
    neg_loss,
    process,
    smoke_process,
    tiny_corpus,
)

DATA_DIR = ROOT / "data"


def fasttext(*, smoke: bool = False) -> None:
    architecture = "skipgram"
    tmp = tiny_corpus() if smoke else None
    try:
        if smoke:
            assert tmp is not None
            processed, path = smoke_process(
                tmp, build_negative_sampler=True, architecture=architecture
            )
        else:
            path = prepare_ag_news(DATA_DIR)
            processed = process(
                path,
                build_huffman=False,
                build_negative_sampler=True,
                architecture=architecture,
                max_sentences=100,
            )
        log_corpus(processed, Path(path))
        chosen = device()
        model = BOW_FastText(
            embedding_dim=EMBEDDING_DIM,
            vocab=processed.vocab,
            num_buckets=64 if smoke else 2_000_000,
        ).to(chosen)
        fit(
            model,
            processed,
            chosen,
            partial(neg_loss, architecture=architecture),
            with_negatives=True,
            epochs=1 if smoke else None,
            batch_size=2 if smoke else None,
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    fasttext(smoke=True)
