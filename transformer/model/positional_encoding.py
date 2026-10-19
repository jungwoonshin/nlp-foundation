"""Padding-aware sinusoidal positions matching the pinned Revisit-MMT/Fairseq code."""
import math
import torch
from torch import Tensor, nn
from transformer.data.vocab import PAD_ID


class SinusoidalPositionalEncoding(nn.Module):
    """Add concatenated sine/cosine channels, then embedding dropout.

    Non-PAD positions begin at padding_idx + 1. PAD contributes no positions;
    its input embedding is preserved. This differs from paper interleaving.
    """
    positional_encoder: Tensor

    def __init__(self, d_model: int, max_positions: int = 1024,
                 dropout: float = .3, padding_idx: int = PAD_ID):
        super().__init__()
        if d_model < 4 or max_positions < 1:
            raise ValueError("reference sinusoids require d_model >= 4 and max_positions > 0")
        self.padding_idx, self.max_positions = padding_idx, max_positions
        half = d_model // 2
        frequencies = torch.exp(torch.arange(half, dtype=torch.float32)
                                * (-math.log(10000.) / (half - 1)))
        positions = torch.arange(max_positions + padding_idx + 1, dtype=torch.float32)
        angles = positions[:, None] * frequencies[None, :]
        pe = torch.cat((angles.sin(), angles.cos()), dim=1)
        if d_model % 2:
            pe = torch.cat((pe, torch.zeros(len(positions), 1)), dim=1)
        pe[padding_idx].zero_()
        self.register_buffer("positional_encoder", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        non_pad = (~padding_mask).long()
        ids = non_pad.cumsum(dim=1) * non_pad + self.padding_idx
        return self.dropout(x + self.positional_encoder[ids].to(x.dtype))
