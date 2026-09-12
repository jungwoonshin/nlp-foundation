from dataclasses import dataclass


@dataclass(frozen=True)
class TextCNNConfig:
    """Hyperparameters for corpus prep and the Kim (2014) CNN-rand setup."""

    filter_sizes: tuple[int, ...] = (3, 4, 5)
    num_filters: int = 100
    dropout: float = 0.5
    embedding_dim: int = 50
    max_length: int = 100
    min_count: int = 5
    seed: int = 42
    max_sentences: int | None = None
    max_examples: int | None = None

    def validate(self) -> None:
        if not self.filter_sizes:
            raise ValueError("filter_sizes must be non-empty")
        if any(size < 1 for size in self.filter_sizes):
            raise ValueError("each filter size must be >= 1")
        if self.num_filters < 1:
            raise ValueError("num_filters must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        if self.max_length < max(self.filter_sizes):
            raise ValueError("max_length must be >= max(filter_sizes)")
        if self.min_count < 1:
            raise ValueError("min_count must be >= 1")
        if self.max_sentences is not None and self.max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")
        if self.max_examples is not None and self.max_examples < 1:
            raise ValueError("max_examples must be >= 1")
