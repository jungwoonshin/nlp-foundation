"""Embedding signatures: paper section 3.4; reference shares all three weight matrices."""
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """tokens (B,L) -> sqrt(d_model) * embeddings (B,L,d_model); PAD row is zero."""
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 1):
        super().__init__()
        raise NotImplementedError("Implement Transformer token embeddings and scaling")

    def forward(self, tokens: Tensor) -> Tensor:
        raise NotImplementedError
