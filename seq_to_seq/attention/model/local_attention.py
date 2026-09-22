from __future__ import annotations

import torch
from torch import nn

from seq_to_seq.attention.config import AlignmentScore
from seq_to_seq.attention.model.types import AttentionOutput


class LocalAttention(nn.Module):
    """Attend inside a window around p_t (Luong et al. 2015, §3.2).

    local-m (monotonic): p_t = t
    local-p (predictive): p_t = S * sigmoid(v_p^T tanh(W_p h_t))
      S is the source length (per row)
    window: [p_t - D, p_t + D] with D = window_size (paper D=10)
    Gaussian: exp(-(s - p_t)^2 / (2 sigma^2)), sigma = D / 2
    Then the same c_t / h_tilde as global attention inside that window.
    """

    def __init__(
        self,
        hidden_size: int,
        score: AlignmentScore,
        *,
        predictive: bool,
        window_size: int = 10,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.score = score
        self.predictive = predictive
        self.window_size = window_size
        raise NotImplementedError("Implement LocalAttention (Luong et al. 2015).")

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
        step: int,
    ) -> AttentionOutput:
        """step is the 0-based decoder timestep used by local-m (p_t = t)."""
        raise NotImplementedError
