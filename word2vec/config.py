from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessingConfig:
    """Hyperparameters that only affect corpus preparation, not the model."""

    min_count: int = 5
    window_size: int = 5
    subsample_threshold: float = 1e-3
    num_negatives: int = 5
    unigram_power: float = 0.75
    negative_table_size: int = 1_000_000
    seed: int = 42
    build_huffman: bool = False
    build_negative_sampler: bool = False

    def validate(self) -> None:
        if self.min_count < 1:
            raise ValueError("min_count must be >= 1")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 0.0 < self.subsample_threshold <= 1.0:
            raise ValueError("subsample_threshold must be in (0, 1]")
        if self.num_negatives < 1:
            raise ValueError("num_negatives must be >= 1")
        if self.unigram_power <= 0:
            raise ValueError("unigram_power must be > 0")
        if self.negative_table_size < 1:
            raise ValueError("negative_table_size must be >= 1")


def resolve_corpus_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Corpus not found: {resolved}")
    return resolved
