from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AlignmentScore = Literal["dot", "general", "concat", "location"]
AttentionKind = Literal["global", "local_m", "local_p"]
OptimizerName = Literal["sgd", "adam"]

ALIGNMENT_SCORES: tuple[AlignmentScore, ...] = ("dot", "general", "concat", "location")
ATTENTION_KINDS: tuple[AttentionKind, ...] = ("global", "local_m", "local_p")
OPTIMIZERS: tuple[OptimizerName, ...] = ("sgd", "adam")


@dataclass(frozen=True)
class LuongConfig:
    """Hyperparameters for Luong et al. (2015) attention-based NMT.

    Pedagogical defaults stay small (2-layer 128-d). Paper WMT models used
    4 layers of 1000 cells, SGD lr=1.0, batch 128, and dropout 0.2.
    """

    attention: AttentionKind = "global"
    score: AlignmentScore = "dot"
    input_feeding: bool = True
    reverse_source: bool = True
    max_len: int = 50
    src_lang: str = "en"
    tgt_lang: str = "vi"
    num_layers: int = 2
    embed_dim: int = 128
    hidden_size: int = 128
    dropout: float = 0.2
    window_size: int = 10
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1.0
    optimizer: OptimizerName = "sgd"
    lr_decay_start: int = 5
    grad_clip: float = 5.0
    init_range: float = 0.1
    seed: int = 42
    max_train_examples: int | None = None
    max_eval_examples: int | None = None

    def validate(self) -> None:
        if self.attention not in ATTENTION_KINDS:
            raise ValueError(f"attention must be one of {ATTENTION_KINDS}")
        if self.score not in ALIGNMENT_SCORES:
            raise ValueError(f"score must be one of {ALIGNMENT_SCORES}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if self.max_len < 1:
            raise ValueError("max_len must be >= 1")
        if not self.src_lang:
            raise ValueError("src_lang must be non-empty")
        if not self.tgt_lang:
            raise ValueError("tgt_lang must be non-empty")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.lr_decay_start < 1:
            raise ValueError("lr_decay_start must be >= 1")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be > 0")
        if self.init_range <= 0.0:
            raise ValueError("init_range must be > 0")
        if self.max_train_examples is not None and self.max_train_examples < 1:
            raise ValueError("max_train_examples must be >= 1")
        if self.max_eval_examples is not None and self.max_eval_examples < 1:
            raise ValueError("max_eval_examples must be >= 1")

    @staticmethod
    def smoke() -> LuongConfig:
        """Tiny Adam run so DummyNMT can overfit without SGD lr=1."""
        config = LuongConfig(
            num_layers=1,
            embed_dim=32,
            hidden_size=32,
            dropout=0.0,
            batch_size=2,
            epochs=1,
            learning_rate=1e-3,
            optimizer="adam",
            lr_decay_start=100,
            reverse_source=True,
            max_len=50,
        )
        config.validate()
        return config
