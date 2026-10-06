"""Sinusoidal position signatures: section 3.5, with padding-aware reference coordinates."""
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add sinusoidal positions to (B,L,d_model).

    Paper: PE(pos,2i)=sin(pos/10000**(2i/d_model)), odd channels use cosine.
    The small-data reference uses Fairseq sinusoidal channel layout/frequencies,
    non-PAD cumulative positions plus padding_idx, and zero vectors at PAD.
    Implement the pinned reference convention for multi30k_tiny, documenting
    its difference from the paper's interleaved layout. No learned positions.
    """
    def __init__(self, d_model: int, max_positions: int = 1024, dropout: float = .3):
        super().__init__()
        raise NotImplementedError("Implement sinusoidal positional encoding")

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        """padding_mask: (B,L), True at PAD; output has the shape of x."""
        raise NotImplementedError
