"""Decoder signatures: masked self-attention, cross-attention, FFN, each post-LN."""
from torch import Tensor, nn
from transformer.config import TransformerConfig


class DecoderLayer(nn.Module):
    """Section 3.1's three post-LN sublayers; preserve (B,T,d_model)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError("Implement Transformer decoder layer")

    def forward(self, x: Tensor, memory: Tensor, self_mask: Tensor, source_mask: Tensor) -> Tensor:
        """x=(B,T,d_model), memory=(B,S,d_model); return shape of x.

        self_mask=(B,1,T,T); source_mask=(B,1,1,S), both True for blocked keys.
        Cross-attention Q comes from decoder, K/V from encoder memory.
        """
        raise NotImplementedError


class Decoder(nn.Module):
    """Independent decoder layers with causal and encoder key-padding masks."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError("Implement Transformer decoder stack")

    def forward(self, x: Tensor, memory: Tensor, self_mask: Tensor, source_mask: Tensor) -> Tensor:
        """Stack decoder_layers independent layers, returning (B,T,d_model)."""
        raise NotImplementedError
