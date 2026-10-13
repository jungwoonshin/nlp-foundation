"""Attention signatures for paper equations 1 and 2; no implemented backend."""
from torch import Tensor, nn
from transformer.model.types import AttentionOutput

import math, torch

def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor,
                                 blocked_mask: Tensor | None = None) -> AttentionOutput:
    """softmax(QK^T/sqrt(d_k))V, masking before softmax.

    q=(B,h,Q,d_k), k=(B,h,K,d_k), v=(B,h,K,d_v).
    blocked_mask broadcasts to (B,h,Q,K); True means blocked. Return context
    (B,h,Q,d_v) and weights (B,h,Q,K). All-blocked queries must not produce NaN.
    """

    sqrt_d_k = math.sqrt(k.shape[-1])
    scores = q @ k.transpose(-2, -1) / sqrt_d_k # (B, h, Q, K)

    if blocked_mask is not None:
        scores = scores.masked_fill(blocked_mask, float("-inf"))
        all_blocked = blocked_mask.all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_blocked, 0.0)

    weights = torch.softmax(scores, dim=-1)# (B, h, Q, K)

    if blocked_mask is not None:
        weights = weights.masked_fill(blocked_mask, 0.0)

    context = weights @ v # (B, h, Q, d_v)
    return AttentionOutput(context=context, weights=weights)


class MultiHeadAttention(nn.Module):
    """Learn Q/K/V projections, concatenate heads, and project through W_O.

    Query/key/value: (B,L,d_model). Return context (B,Q,d_model), per-head
    weights (B,h,Q,K). Match reference linear biases and initialization;
    attention dropout is zero in the Multi30k reference profile.
    """
    def __init__(self, d_model: int, heads: int, dropout: float = 0.):
        super().__init__()

        self.heads = heads
        self.d_k = d_model // heads
        self.d_v = d_model // heads

        self.q_proj = torch.nn.Linear(d_model, heads * self.d_k, bias=False)
        self.k_proj = torch.nn.Linear(d_model, heads * self.d_k, bias=False)
        self.v_proj = torch.nn.Linear(d_model, heads * self.d_v, bias=False)
        self.o_proj = torch.nn.Linear(heads * self.d_v, d_model, bias=False)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                blocked_mask: Tensor | None = None) -> AttentionOutput:

        query, key, value = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        # query/key is (B,Q/K,heads * self.d_k), and value is (B,K,heads * self.d_v)
        B, Q, _ = query.shape
        K = key.size(1)
        h = self.heads

        q = query.reshape(B, Q, h, self.d_k).transpose(1, 2)
        k = key.reshape(B, K, h, self.d_k).transpose(1, 2)
        v = value.reshape(B, K, h, self.d_v).transpose(1, 2)

        attention_output = scaled_dot_product_attention(q, k, v, blocked_mask)
        context = attention_output.context.transpose(1, 2)
        context = context.reshape(B, Q, h * self.d_v)
        context = self.o_proj(context)

        return AttentionOutput(
            context=context,                   # (B, Q, d_model)
            weights=attention_output.weights,  # (B, h, Q, K)
        )
        
     
