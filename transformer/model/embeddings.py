"""Embedding signatures: paper section 3.4; reference shares all three weight matrices."""
from torch import Tensor, nn
import torch
import math

class TokenEmbedding(nn.Module):
    """tokens (B,L) -> sqrt(d_model) * embeddings (B,L,d_model); PAD row is zero."""
    def __init__(self, vocab_size: int, d_model: int = 512, padding_idx: int = 1):
        super().__init__()
        self.D_MODEL = d_model
        self.embedding = nn.Embedding(vocab_size, self.D_MODEL, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, mean=0., std=d_model ** -0.5)
        with torch.no_grad():
            self.embedding.weight[padding_idx].zero_()
        self.NORMALIZE_CONSTANT = math.sqrt(self.D_MODEL)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens) * self.NORMALIZE_CONSTANT
