"""Train skip-gram or CBOW with negative sampling."""

from __future__ import annotations

from functools import partial

from word2vec.negative_sampling.model import NegativeSampling
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


def negative_sampling(architecture: str = "skipgram", *, smoke: bool = False) -> None:
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
        model = NegativeSampling(
            embedding_dim=EMBEDDING_DIM,
            vocab_size=len(processed.vocab),
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
    negative_sampling(smoke=True)
