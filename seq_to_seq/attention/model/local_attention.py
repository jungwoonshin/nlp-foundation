from __future__ import annotations

import torch
from torch import nn

from seq_to_seq.attention.config import AttentionKind
from seq_to_seq.attention.model.global_attention import GlobalAttention
from seq_to_seq.attention.model.types import AttentionOutput


class LocalAttention(nn.Module):
    """Local attention with general scoring (Luong et al. 2015, §3.2).

    Centers use zero-based original-source coordinates. Local-m clamps p_t=t
    to the final source token. Local-p anchors its 2D+1 window at floor(p_t).
    window_size is the radius D (paper D=10), not the full window width.
    Content scores are normalized over valid positions inside the window.
    Local-p then multiplies them by a Gaussian centered on its prediction.
    Empty sources return zero context and weights.
    Local-p predicts a separate floating-point center for each example.
    """

    def __init__(
        self,
        hidden_size: int,
        attention_kind: AttentionKind,
        *,
        window_size: int = 10,
        reverse_source: bool = False,
    ) -> None:
        super().__init__()
        if attention_kind not in ("local_m", "local_p"):
            raise NotImplementedError("Only local_m and local_p attention are supported.")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.attention_kind = attention_kind
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.reverse_source = reverse_source
        self.global_attention = GlobalAttention(
            hidden_size, score="general", max_source_length=2 * window_size + 1
        )
        if attention_kind == "local_p":
            self.weight_p = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight_va = nn.Linear(hidden_size, 1, bias=False)
            self.gaussian_attention = GlobalAttention(
                hidden_size, score="general", max_source_length=2 * window_size + 1, is_gaussian=True
            )

    def monotonic_p_t(self, time_point: int) -> int:
        if time_point < 0:
            raise ValueError("step must be >= 0")
        return time_point

    def predictive_p_t(self, decoder_hidden: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        # decoder_hidden: (batch, hidden)
        activated = torch.tanh(self.weight_p(decoder_hidden))  # (batch, hidden)
        probability = torch.sigmoid(self.weight_va(activated)).squeeze(-1)  # (B,)
        p_t = src_lengths.to(decoder_hidden.device) * probability  # (B,)
        return p_t

    def get_start_end(self, center_point: int, source_len: int) -> tuple[int, int]:
        """Return shared, clipped [start, end) bounds in the padded source."""
        start = min(max(center_point - self.window_size, 0), source_len)
        end = min(center_point + self.window_size + 1, source_len)
        return start, end

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_lengths: torch.Tensor,
        step: int,
    ) -> AttentionOutput:
        """step is the 0-based timestep used by local-m; local-p predicts p_t."""

        if encoder_outputs.dim() != 3:
            raise ValueError("encoder_outputs must have shape (batch, source_len, hidden)")
        batch, source_len, _ = encoder_outputs.shape
        if src_lengths.shape != (batch,):
            raise ValueError("src_lengths must have shape (batch,)")
        if bool(((src_lengths < 0) | (src_lengths > source_len)).any()):
            raise ValueError(
                f"src_lengths must be between 0 and padded source length {source_len}"
            )

        lengths = src_lengths.to(encoder_outputs.device)
        if self.attention_kind == "local_p":
            p_t = self.predictive_p_t(decoder_hidden, src_lengths)
            anchor = p_t.detach().floor().long()
            attention = self.gaussian_attention
        else:
            p_t = torch.minimum(
                torch.full_like(lengths, self.monotonic_p_t(step)), (lengths - 1).clamp(min=0)
            )
            anchor = p_t
            attention = self.global_attention
        # Official one-based mu=L*sigmoid(...)+1 becomes zero-based p_t here.
        centers = p_t
        if self.reverse_source:
            anchor = lengths - 1 - anchor
            centers = lengths - 1 - p_t
        if source_len == 0:
            result = attention(decoder_hidden, encoder_outputs, lengths, centers)
            return AttentionOutput(context=result.context, weights=result.weights, p_t=p_t)
        start = (anchor - self.window_size).clamp(0, source_len)
        end = torch.minimum(anchor + self.window_size + 1, lengths)
        offsets = torch.arange(min(source_len, 2 * self.window_size + 1), device=encoder_outputs.device)
        indices = (start.unsqueeze(1) + offsets).clamp(max=source_len - 1)
        local_encoder = encoder_outputs.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )
        result = attention(
            decoder_hidden, local_encoder, (end - start).clamp(min=0), centers,
            source_start=start.unsqueeze(1),
        )
        weights = encoder_outputs.new_zeros(batch, source_len).scatter_add(1, indices, result.weights)
        return AttentionOutput(context=result.context, weights=weights, p_t=p_t)
