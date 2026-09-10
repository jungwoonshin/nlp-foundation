"""Train skip-gram FastText (subword NEG) on the word2vec corpus pipeline."""

from __future__ import annotations

from functools import partial

from word2vec.subword.model import SubwordNegativeSampling
from word2vec.training import (
    DEFAULT_CORPUS,
    EMBEDDING_DIM,
    device,
    fit,
    log_corpus,
    neg_loss,
    process,
    smoke_process,
    tiny_corpus,
)


def subword_negative_sampling(architecture: str = "skipgram", *, smoke: bool = False) -> None:
    if architecture != "skipgram":
        raise ValueError("Subword NEG only supports skipgram; CBOW bags are not encoded yet.")
    tmp = tiny_corpus() if smoke else None
    try:
        if smoke:
            assert tmp is not None
            processed, path = smoke_process(
                tmp, build_negative_sampler=True, architecture=architecture
            )
        else:
            processed = process(
                DEFAULT_CORPUS,
                build_huffman=False,
                build_negative_sampler=True,
                architecture=architecture,
            )
            path = DEFAULT_CORPUS
        log_corpus(processed, path)
        chosen = device()
        model = SubwordNegativeSampling(
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
    subword_negative_sampling(smoke=True)
