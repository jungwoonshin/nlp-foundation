from __future__ import annotations

import torch
from torch import nn

from seq_to_seq.attention.config import AttentionKind
from seq_to_seq.attention.model.global_attention import GlobalAttention
from seq_to_seq.attention.model.types import AttentionOutput


class LocalAttention(nn.Module):
    """Monotonic local attention with general scoring (Luong et al. 2015, §3.2).

    The center is p_t = t, with an inclusive window [t - D, t + D].
    window_size is the radius D (paper D=10), not the full window width.
    Scores are normalized over valid source positions inside the window.
    Empty windows return zero context and weights, without shifting p_t.
    Predictive local-p attention is not implemented.
    """

    def __init__(
        self,
        hidden_size: int,
        attention_kind: AttentionKind,
        *,
        window_size: int = 10,
    ) -> None:
        super().__init__()
        if attention_kind != "local_m":
            raise NotImplementedError("Only local_m attention is implemented.")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.global_attention = GlobalAttention(
            hidden_size, score="general", max_source_length=2 * window_size + 1
        )

    def monotonic_p_t(self, time_point: int) -> int:
        if time_point < 0:
            raise ValueError("step must be >= 0")
        return time_point

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
        """step is the 0-based decoder timestep used by local-m (p_t = t)."""

        if encoder_outputs.dim() != 3:
            raise ValueError("encoder_outputs must have shape (batch, source_len, hidden)")
        batch, source_len, _ = encoder_outputs.shape
        if src_lengths.shape != (batch,):
            raise ValueError("src_lengths must have shape (batch,)")
        if bool(((src_lengths < 0) | (src_lengths > source_len)).any()):
            raise ValueError(
                f"src_lengths must be between 0 and padded source length {source_len}"
            )

        p_t = self.monotonic_p_t(step)
        start, end = self.get_start_end(p_t, source_len)
        local_encoder = encoder_outputs[:, start:end, :]
        local_lengths = (src_lengths - start).clamp(min=0, max=end - start)
        local_attention = self.global_attention(
            decoder_hidden, local_encoder, local_lengths
        )
        # Preserve source coordinates for callers and attention visualizations.
        weights = torch.nn.functional.pad(
            local_attention.weights, (start, source_len - end)
        )
        centers = torch.full((batch,), p_t, dtype=torch.long, device=encoder_outputs.device)
        return AttentionOutput(
            context=local_attention.context, weights=weights, p_t=centers
        )
