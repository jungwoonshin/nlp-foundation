from __future__ import annotations

import torch
from torch import nn

from seq_to_seq.attention.config import ALIGNMENT_SCORES, AlignmentScore
from seq_to_seq.attention.model.types import AttentionOutput


class GlobalAttention(nn.Module):
    """Attend over every source position (Luong et al. 2015, §3.1).

    Content scores compare the current decoder state with every encoder state.
    The location score projects the decoder state to a fixed source-position
    capacity. PAD positions are masked before normalizing the scores.
    """

    def __init__(
        self,
        hidden_size: int,
        score: AlignmentScore,
        max_source_length: int,
    ) -> None:
        super().__init__()
        if score not in ALIGNMENT_SCORES:
            raise ValueError(f"score must be one of {ALIGNMENT_SCORES}")
        if max_source_length < 1:
            raise ValueError("max_source_length must be >= 1")
        self.hidden_size = hidden_size
        self.score = score
        self.softmax = nn.Softmax(dim=-1)
        self.max_source_length = max_source_length
        self.weight_a: nn.Linear | None = None
        self.v_a: nn.Linear | None = None
        self.activation: nn.Tanh | None = None

        if score == "general":
            self.weight_a = nn.Linear(hidden_size, hidden_size, bias=False)
        elif score == "concat":
            self.weight_a = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.activation = nn.Tanh()
            self.v_a = nn.Linear(hidden_size, 1, bias=False)
        elif score == "location":
            # Equation (9): a_t = softmax(W_a h_t), without a bias term.
            self.weight_a = nn.Linear(hidden_size, max_source_length, bias=False)

    def dot_score(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Dot alignment: score(h_t, h_s) = h_t^T h_s.

        decoder_hidden: (batch, hidden)
        encoder_outputs: (batch, source_len, hidden)
        returns: (batch, source_len)
        """
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(-1))
        return scores.squeeze(-1)

    def general_score(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """General alignment: score(h_t, h_s) = h_t^T W_a h_s.

        decoder_hidden: (batch, hidden)
        encoder_outputs: (batch, source_len, hidden)
        weight_a: (hidden, hidden)
        returns: (batch, source_len)
        """
        if self.weight_a is None:
            raise RuntimeError("general score requires weight_a")
        projected_decoder = decoder_hidden @ self.weight_a.weight
        scores = torch.bmm(encoder_outputs, projected_decoder.unsqueeze(-1))
        return scores.squeeze(-1)

    def concat_score(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Concat alignment: score = v_a^T tanh(W_a [h_t ; h_s]).

        decoder_hidden: (batch, hidden)
        encoder_outputs: (batch, source_len, hidden)
        weight_a: (hidden, 2 * hidden)
        v_a: (hidden,)
        returns: (batch, source_len)
        """
        if self.weight_a is None or self.activation is None or self.v_a is None:
            raise RuntimeError("concat score requires weight_a, tanh, and v_a")
        decoder_by_source = decoder_hidden.unsqueeze(1).expand(
            -1, encoder_outputs.size(1), -1
        )
        combined = torch.cat(
            [decoder_by_source, encoder_outputs],
            dim=-1,
        )
        activated = self.activation(self.weight_a(combined))
        return self.v_a(activated).squeeze(-1)

    def location_score(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Location alignment: score = W_a h_t, independent of source content.

        decoder_hidden: (batch, hidden)
        encoder_outputs: (batch, source_len, hidden); only its length is used
        weight_a: (max_source_length, hidden)
        returns: (batch, source_len)
        """
        source_len = encoder_outputs.size(1)
        if source_len > self.max_source_length:
            raise ValueError(
                f"padded source length {source_len} exceeds location-attention "
                f"capacity {self.max_source_length}"
            )
        if self.weight_a is None:
            raise RuntimeError("location score requires weight_a")
        return self.weight_a(decoder_hidden)[:, :source_len]

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> AttentionOutput:
        """Compute one decoder step of global attention.

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
        score_functions = {
            "location": self.location_score,
            "general": self.general_score,
            "concat": self.concat_score,
            "dot": self.dot_score,
        }
        try:
            score_function = score_functions[self.score]
        except KeyError as error:
            raise ValueError(f"Invalid score: {self.score}") from error

        attention_scores = score_function(decoder_hidden, encoder_outputs)
        lengths = src_lengths.to(device=decoder_hidden.device)
        if bool((lengths > source_len).any()):
            raise ValueError(
                f"src_lengths cannot exceed padded source length {source_len}"
            )
        positions = torch.arange(source_len, device=decoder_hidden.device)
        padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
        masked_scores = attention_scores.masked_fill(padding_mask, -torch.inf)
        weights = self.softmax(masked_scores)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return AttentionOutput(context=context, weights=weights)
