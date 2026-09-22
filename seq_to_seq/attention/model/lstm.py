from __future__ import annotations

import torch
from torch import nn

from seq_to_seq.attention.model.types import LSTMState


class LSTMCell(nn.Module):
    """One LSTM step with PyTorch gate order: input, forget, cell, output.

    h_t, c_t = LSTM(x_t, h_{t-1}, c_{t-1})
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        raise NotImplementedError("Implement LSTMCell (Luong et al. 2015).")

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x_t, h_prev, c_prev: (batch, *) -> h_t, c_t each (batch, hidden)."""
        raise NotImplementedError


class StackedLSTM(nn.Module):
    """Unrolled stack of LSTMCell. Layer 0 reads x; layer i reads layer i-1.

    At padded timesteps (t >= length), keep the previous state and write zeros
    to the output. Dropout (Zaremba et al. 2015) belongs between layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        raise NotImplementedError("Implement StackedLSTM (Luong et al. 2015).")

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        state: LSTMState | None = None,
    ) -> tuple[torch.Tensor, LSTMState]:
        """x: (batch, seq_len, input) -> outputs (batch, seq_len, hidden), state.

        state h/c each (num_layers, batch, hidden).
        """
        raise NotImplementedError
