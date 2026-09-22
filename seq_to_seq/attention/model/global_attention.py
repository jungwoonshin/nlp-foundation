from __future__ import annotations

import torch
from torch import nn

from seq_to_seq.attention.config import AlignmentScore
from seq_to_seq.attention.model.types import AttentionOutput


class GlobalAttention(nn.Module):
    """Attend over every source position (Luong et al. 2015, §3.1).

    alpha_t(s) = softmax(score(h_t, h_s))  with PAD positions masked to -inf
    c_t = sum_s alpha_t(s) h_s
    h_tilde_t = tanh(W_c [c_t ; h_t])
    """

    def __init__(self, hidden_size: int, score: AlignmentScore) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.score = score
        raise NotImplementedError("Implement GlobalAttention (Luong et al. 2015).")

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> AttentionOutput:
        """decoder_hidden: (batch, hidden); encoder_outputs: (batch, src_len, hidden).

        source_mask: (batch, src_len) True on real tokens, False on PAD.
        """
        raise NotImplementedError
