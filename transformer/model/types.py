"""Shape-carrying model results. Masks use True for blocked connections."""
from dataclasses import dataclass
from torch import Tensor


@dataclass
class AttentionOutput:
    context: Tensor  # (B,heads,Q,d_v) for scaled attention, (B,Q,d_model) after projection
    weights: Tensor  # (B,heads,Q,K)
