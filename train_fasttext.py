"""Build a FastText-paper corpus (AG News) with the word2vec pipeline."""

from __future__ import annotations

from pathlib import Path

from prepare_ag_news import prepare_ag_news
from train_word2vec import _log_corpus, process

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def fasttext() -> None:
    corpus = prepare_ag_news(DATA_DIR)
    processed = process(
        corpus,
        build_huffman=False,
        build_negative_sampler=False,
        architecture="skipgram",
        max_sentences=100,
    )
    _log_corpus(processed, corpus)


if __name__ == "__main__":
    fasttext()
