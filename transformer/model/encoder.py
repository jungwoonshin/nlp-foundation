"""Encoder signatures: self-attention then position-wise FFN, each with post-LN."""
from torch import Tensor, nn
from transformer.config import TransformerConfig


class EncoderLayer(nn.Module):
    """Self-attention then FFN, each post-LN; preserve (B,S,d_model)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError("Implement Transformer encoder layer")

    def forward(self, x: Tensor, blocked_mask: Tensor) -> Tensor:
        """x=(B,S,d_model); blocked_mask=(B,1,1,S); output matches x."""
        raise NotImplementedError


class Encoder(nn.Module):
    """Section 3.1 encoder stack; no parameter sharing between layers."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError("Implement Transformer encoder stack")

    def forward(self, x: Tensor, blocked_mask: Tensor) -> Tensor:
        """Stack encoder_layers independent layers, returning (B,S,d_model)."""
        raise NotImplementedError
