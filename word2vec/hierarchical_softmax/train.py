"""Train skip-gram or CBOW with hierarchical softmax."""

from __future__ import annotations

from functools import partial

from word2vec.hierarchical_softmax.model import HierarchicalSoftmax
from word2vec.training import (
    DEFAULT_CORPUS,
    EMBEDDING_DIM,
    device,
    fit,
    hs_loss,
    log_corpus,
    process,
    smoke_process,
    tiny_corpus,
)


def hierarchical_softmax(architecture: str = "skipgram", *, smoke: bool = False) -> None:
    tmp = tiny_corpus() if smoke else None
    try:
        if smoke:
            assert tmp is not None
            processed, path = smoke_process(
                tmp, build_huffman=True, architecture=architecture
            )
        else:
            processed = process(
                DEFAULT_CORPUS,
                build_huffman=True,
                build_negative_sampler=False,
                architecture=architecture,
            )
            path = DEFAULT_CORPUS
        if processed.vocab.coding is None:
            raise RuntimeError("Huffman codes are required for hierarchical softmax.")
        log_corpus(processed, path)
        chosen = device()
        model = HierarchicalSoftmax(
            processed.vocab.coding,
            embedding_dim=EMBEDDING_DIM,
            vocab_size=len(processed.vocab),
        ).to(chosen)
        fit(
            model,
            processed,
            chosen,
            partial(hs_loss, architecture=architecture),
            with_negatives=False,
            epochs=1 if smoke else None,
            batch_size=2 if smoke else None,
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    hierarchical_softmax(smoke=True)
