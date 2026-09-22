from __future__ import annotations

import torch


def score_dot(h_t: torch.Tensor, source_h: torch.Tensor) -> torch.Tensor:
    """Dot alignment: score(h_t, h_s) = h_t^T h_s.

    h_t: (batch, hidden)
    source_h: (batch, source_len, hidden)
    returns: (batch, source_len)
    """
    raise NotImplementedError


def score_general(
    h_t: torch.Tensor,
    source_h: torch.Tensor,
    weight_a: torch.Tensor,
) -> torch.Tensor:
    """General (bilinear) alignment: score = h_t^T W_a h_s.

    weight_a: (hidden, hidden)
    returns: (batch, source_len)
    """
    raise NotImplementedError


def score_concat(
    h_t: torch.Tensor,
    source_h: torch.Tensor,
    weight_a: torch.Tensor,
    v_a: torch.Tensor,
) -> torch.Tensor:
    """Concat alignment: score = v_a^T tanh(W_a [h_t ; h_s]).

    weight_a: (hidden, 2 * hidden) or equivalent
    v_a: (hidden,)
    returns: (batch, source_len)
    """
    raise NotImplementedError


def score_location(h_t: torch.Tensor, source_len: int, weight_a: torch.Tensor) -> torch.Tensor:
    """Location alignment: score = W_a h_t, independent of source content.

    weight_a maps hidden -> source_len (paper: a vector of scores per step)
    h_t: (batch, hidden)
    returns: (batch, source_len)
    """
    raise NotImplementedError
