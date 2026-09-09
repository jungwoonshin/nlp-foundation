"""Train skip-gram FastText (subword NEG) on the word2vec corpus pipeline."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from train_word2vec import _device, _fit, _log_corpus, _neg_loss, process
from word2vec.negative_sampling import SubwordNegativeSampling

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "data" / "text8m1.txt"
EMBEDDING_DIM = 24
BATCH_SIZE = 256
LEARNING_RATE = 0.025
EPOCHS = 200


def subword_negative_sampling(architecture: str = "skipgram") -> None:
    if architecture != "skipgram":
        raise ValueError("Subword NEG only supports skipgram; CBOW bags are not encoded yet.")
    processed = process(
        DEFAULT_CORPUS,
        build_huffman=False,
        build_negative_sampler=True,
        architecture=architecture,
    )
    _log_corpus(processed)
    device = _device()
    model = SubwordNegativeSampling(
        embedding_dim=EMBEDDING_DIM,
        vocab=processed.vocab,
    ).to(device)
    _fit(
        model,
        processed,
        device,
        partial(_neg_loss, architecture=architecture),
        with_negatives=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )


if __name__ == "__main__":
    subword_negative_sampling()
