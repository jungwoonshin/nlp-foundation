"""Attention signatures for paper equations 1 and 2; no implemented backend."""
from torch import Tensor, nn
from transformer.model.types import AttentionOutput


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor,
                                 blocked_mask: Tensor | None = None) -> AttentionOutput:
    """softmax(QK^T/sqrt(d_k))V, masking before softmax.

    q=(B,h,Q,d_k), k=(B,h,K,d_k), v=(B,h,K,d_v).
    blocked_mask broadcasts to (B,h,Q,K); True means blocked. Return context
    (B,h,Q,d_v) and weights (B,h,Q,K). All-blocked queries must not produce NaN.
    """
    raise NotImplementedError("Implement scaled dot-product attention")


class MultiHeadAttention(nn.Module):
    """Learn Q/K/V projections, concatenate heads, and project through W_O.

    Query/key/value: (B,L,d_model). Return context (B,Q,d_model), per-head
    weights (B,h,Q,K). Match reference linear biases and initialization;
    attention dropout is zero in the Multi30k reference profile.
    """
    def __init__(self, d_model: int, heads: int, dropout: float = 0.):
        super().__init__()
        raise NotImplementedError("Implement multi-head attention")

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                blocked_mask: Tensor | None = None) -> AttentionOutput:
        raise NotImplementedError
