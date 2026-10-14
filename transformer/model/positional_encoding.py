"""Sinusoidal position signatures: section 3.5, with padding-aware reference coordinates."""
from torch import Tensor, nn

import torch

class SinusoidalPositionalEncoding(nn.Module):
    """Add sinusoidal positions to (B,L,d_model).

    Paper: PE(pos,2i)=sin(pos/10000**(2i/d_model)), odd channels use cosine.
    The small-data reference uses Fairseq sinusoidal channel layout/frequencies,
    non-PAD cumulative positions plus padding_idx, and zero vectors at PAD.
    Implement the pinned reference convention for multi30k_tiny, documenting
    its difference from the paper's interleaved layout. No learned positions.
    """

    positional_encoder: Tensor

    def __init__(self, d_model: int, max_positions: int = 1024, dropout: float = .3):
        super().__init__()
        # (pos, i)
        pos = torch.arange(max_positions, dtype=torch.float32)[:, None]  # (POS, 1)
        dims = torch.arange(0, d_model, 2, dtype=torch.float32)[None, :] # (1, D_MODEL//2)
        # i = torch.arange(d_model, dtype=torch.float32)[None, :]      # (1, I)
        angles = pos / (10000.0 ** (dims / d_model)) # (POS, I)

        pe = torch.empty(max_positions, d_model, device=angles.device, dtype=angles.dtype)
        pe[:, 0::2] = torch.sin(angles)  # dimensions 0, 2, 4, ...
        pe[:, 1::2] = torch.cos(angles[:, :d_model // 2])  # dimensions 1, 3, 5, ...
        self.register_buffer("positional_encoder", pe)
        self.dropout = nn.Dropout(dropout)

    # x: (B,L,d_model)
    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        """padding_mask: (B,L), True at PAD; zero the full PAD vectors.

        Apply dropout to embeddings plus positions; output has the shape of x.
        """

        # padding_mask: (B, L), True at PAD
        non_pad = (~padding_mask).long()
        position_ids = (non_pad.cumsum(dim=1) - 1).clamp_min(0)

        positions = self.positional_encoder[position_ids]
        positions = positions.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return self.dropout(x + positions)
