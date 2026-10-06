"""Boolean masks; True always means an attention connection is blocked."""
import torch
from transformer.data.vocab import PAD_ID


def padding_mask(tokens):
    """(B,S) token ids -> (B,1,1,S) blocked keys, including left padding."""
    return tokens.eq(PAD_ID)[:, None, None, :]


def causal_mask(length, *, device=None):
    """(1,1,T,T); block future keys, not the current key."""
    if length < 1:
        raise ValueError("length must be positive")
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)[None, None]


def decoder_mask(tokens):
    return padding_mask(tokens) | causal_mask(tokens.size(1), device=tokens.device)
