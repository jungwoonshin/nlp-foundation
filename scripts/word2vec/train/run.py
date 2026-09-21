"""Train skip-gram or CBOW with hierarchical softmax or negative sampling."""

from word2vec.hierarchical_softmax.train import hierarchical_softmax
from word2vec.negative_sampling.train import negative_sampling
from word2vec.subword.train import subword_negative_sampling
from word2vec.training import process

__all__ = [
    "hierarchical_softmax",
    "negative_sampling",
    "process",
    "subword_negative_sampling",
]


if __name__ == "__main__":
    negative_sampling()
