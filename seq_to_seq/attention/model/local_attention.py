from __future__ import annotations

import torch
from torch import nn

from seq_to_seq.attention.config import AttentionKind
from seq_to_seq.attention.model.global_attention import GlobalAttention
from seq_to_seq.attention.model.types import AttentionOutput


class LocalAttention(nn.Module):
    """Local attention with general scoring (Luong et al. 2015, §3.2).

    For local-m, the center is p_t = t, with an inclusive window [t - D, t + D].
    window_size is the radius D (paper D=10), not the full window width.
    Content scores are normalized over valid positions inside the window.
    Local-p then multiplies them by a Gaussian centered on its prediction.
    Empty windows return zero context and weights, without shifting p_t.
    Local-p predicts a separate floating-point center for each example.
    """

    def __init__(
        self,
        hidden_size: int,
        attention_kind: AttentionKind,
        *,
        window_size: int = 10,
    ) -> None:
        super().__init__()
        if attention_kind not in ("local_m", "local_p"):
            raise NotImplementedError("Only local_m and local_p attention are supported.")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.attention_kind = attention_kind
        self.hidden_size = hidden_size
        self.window_size = window_size
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

        if self.attention_kind == "local_p":
            p_t = self.predictive_p_t(decoder_hidden, src_lengths)
            if source_len == 0:
                result = self.gaussian_attention(decoder_hidden, encoder_outputs, src_lengths, p_t)
                return AttentionOutput(context=result.context, weights=result.weights, p_t=p_t)
            lengths = src_lengths.to(encoder_outputs.device)
            # Round before integer offsets to preserve boundaries near integers.
            start = (p_t.detach().ceil().long() - self.window_size).clamp(0, source_len)
            end = torch.minimum(
                p_t.detach().floor().long() + self.window_size + 1, lengths
            )
            local_lengths = (end - start).clamp(min=0)
            offsets = torch.arange(
                min(source_len, 2 * self.window_size + 1), device=encoder_outputs.device
            )
            indices = (start.unsqueeze(1) + offsets).clamp(max=source_len - 1)
            local_encoder = encoder_outputs.gather(
                1, indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            )
            result = self.gaussian_attention(
                decoder_hidden, local_encoder, local_lengths, p_t, source_start=start.unsqueeze(1)
            )
            weights = encoder_outputs.new_zeros(batch, source_len).scatter_add(
                1, indices, result.weights
            )
            return AttentionOutput(
                context=result.context, weights=weights, p_t=p_t
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
