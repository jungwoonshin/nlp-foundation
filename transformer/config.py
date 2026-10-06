"""Explicit small-data reference settings, separate from the original WMT run."""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class TransformerConfig:
    profile: str = "multi30k_tiny"
    encoder_layers: int = 4
    decoder_layers: int = 4
    d_model: int = 128
    d_ff: int = 256
    heads: int = 4
    dropout: float = 0.3
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    max_positions: int = 1024
    max_tokens: int = 4096
    batch_size: int | None = None
    accumulation_steps: int = 2
    max_updates: int = 8000
    max_epochs: int = 1000
    patience: int = 10
    label_smoothing: float = 0.1
    adam_betas: tuple[float, float] = (0.9, 0.98)
    adam_eps: float = 1e-8
    learning_rate: float = 0.005
    warmup_init_lr: float = 1e-7
    warmup_updates: int = 2000
    fixed_lr: bool = False
    grad_clip: float = 25.0
    seed: int = 1
    beam_size: int = 5
    length_penalty: float = 1.0
    max_decode_tokens: int = 200

    def validate(self) -> None:
        for name in ("encoder_layers", "decoder_layers", "d_model", "d_ff", "heads",
                     "max_positions", "max_tokens", "accumulation_steps", "max_updates",
                     "max_epochs", "patience", "warmup_updates", "beam_size", "max_decode_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        for name in ("dropout", "attention_dropout", "activation_dropout", "label_smoothing"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0,1)")
        for name in ("learning_rate", "adam_eps", "grad_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.warmup_init_lr <= self.learning_rate:
            raise ValueError("warmup_init_lr must be between zero and learning_rate")
        if len(self.adam_betas) != 2 or any(not 0 <= b < 1 for b in self.adam_betas):
            raise ValueError("adam_betas must contain two values in [0,1)")
        if self.seed < 0 or not math.isfinite(self.length_penalty) or self.length_penalty < 0:
            raise ValueError("seed and length_penalty must be nonnegative")

    @classmethod
    def smoke(cls):
        return replace(cls(), profile="smoke", encoder_layers=1, decoder_layers=1,
                       d_model=16, d_ff=32, heads=2, dropout=0., batch_size=2,
                       accumulation_steps=1, max_updates=100, max_epochs=1,
                       label_smoothing=0., fixed_lr=True, learning_rate=1e-3,
                       warmup_init_lr=0., beam_size=1, max_decode_tokens=10)
