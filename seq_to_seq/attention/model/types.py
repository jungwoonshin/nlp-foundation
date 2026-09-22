from __future__ import annotations

from dataclasses import dataclass

import torch

from seq_to_seq.attention.config import AlignmentScore, AttentionKind

LSTMState = tuple[torch.Tensor, torch.Tensor]


@dataclass
class AttentionOutput:
    """Global or local attention result at one decoder step.

    context: (batch, hidden) weighted source summary c_t
    weights: (batch, source_len) alignment probabilities (padded positions 0)
    p_t: (batch,) local predicted/monotonic centers, or None for global attention
    """

    context: torch.Tensor
    weights: torch.Tensor
    p_t: torch.Tensor | None = None


__all__ = [
    "AlignmentScore",
    "AttentionKind",
    "AttentionOutput",
    "LSTMState",
]
