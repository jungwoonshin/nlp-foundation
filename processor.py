"""Turn a whitespace-tokenized corpus into skip-gram tensors for PyTorch.

This module is the data-processing entry point. It does not define a model.
"""

from __future__ import annotations

from pathlib import Path

from word2vec.config import ProcessingConfig
from word2vec.pipeline import ProcessedCorpus, process_corpus

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "data" / "text8m1.txt"


def process(
    path: str | Path = DEFAULT_CORPUS,
    *,
    min_count: int = 5,
    window_size: int = 5,
    subsample_threshold: float = 1e-3,
    num_negatives: int = 5,
    seed: int = 42,
) -> ProcessedCorpus:
    """Load `path` and return a vocab plus a PyTorch Dataset of skip-gram pairs."""

    config = ProcessingConfig(
        min_count=min_count,
        window_size=window_size,
        subsample_threshold=subsample_threshold,
        num_negatives=num_negatives,
        seed=seed,
    )
    return process_corpus(path, config)


if __name__ == "__main__":
    processed = process()
    batch = next(iter(processed.dataloader(batch_size=4, shuffle=False)))
    print(f"corpus: {DEFAULT_CORPUS}")
    print(f"raw tokens: {processed.raw_token_count:,}")
    print(f"tokens after min_count + subsample: {processed.kept_token_count:,}")
    print(f"vocab size: {len(processed.vocab):,}")
    print(f"skip-gram pairs: {len(processed.dataset):,}")
    print(
        "sample batch shapes:",
        {name: tuple(tensor.shape) for name, tensor in batch.items()},
    )
