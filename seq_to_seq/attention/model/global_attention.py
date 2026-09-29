from __future__ import annotations

import torch
from torch import nn

from seq_to_seq.attention.config import AlignmentScore
from seq_to_seq.attention.model.types import AttentionOutput


class GlobalAttention(nn.Module):
    """Attend over every source position (Luong et al. 2015, §3.1).

    a_t = softmax(W_a h_t), with PAD positions masked to -inf
    c_t = sum_s alpha_t(s) h_s
    """

    def __init__(
        self,
        hidden_size: int,
        score: AlignmentScore,
        max_source_length: int,
    ) -> None:
        super().__init__()
        if score != "location":
            raise NotImplementedError(
                "GlobalAttention currently implements only the location score."
            )
        if max_source_length < 1:
            raise ValueError("max_source_length must be >= 1")
        self.hidden_size = hidden_size
        self.score = score
        self.softmax = nn.Softmax(dim=-1)
        self.max_source_length = max_source_length
        # Equation (9): a_t = softmax(W_a h_t). The paper does not add a bias.
        self.weight_a = nn.Linear(hidden_size, max_source_length, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> AttentionOutput:
        """Compute one decoder step of location-based global attention.

        decoder_hidden: (batch, hidden)
        encoder_outputs: (batch, padded_source_len, hidden)
        src_lengths: (batch,)
        """
        if decoder_hidden.dim() != 2:
            raise ValueError("decoder_hidden must have shape (batch, hidden)")
        if encoder_outputs.dim() != 3:
            raise ValueError(
                "encoder_outputs must have shape (batch, source_len, hidden)"
            )

        source_len = encoder_outputs.size(1)
        if source_len > self.max_source_length:
            raise ValueError(
                f"padded source length {source_len} exceeds location-attention "
                f"capacity {self.max_source_length}"
            )
        source_states = encoder_outputs
        attention_scores = self.weight_a(decoder_hidden)[:, :source_len]
        lengths = src_lengths.to(device=decoder_hidden.device)
        if bool((lengths > source_len).any()):
            raise ValueError(
                f"src_lengths cannot exceed padded source length {source_len}"
            )
        positions = torch.arange(source_len, device=decoder_hidden.device)
        padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
        masked_scores = attention_scores.masked_fill(padding_mask, -torch.inf)
        weights = self.softmax(masked_scores)
        context = torch.bmm(weights.unsqueeze(1), source_states).squeeze(1)
        return AttentionOutput(context=context, weights=weights)
