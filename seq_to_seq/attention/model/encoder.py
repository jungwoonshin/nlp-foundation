from __future__ import annotations

import torch
from torch import nn

from seq_to_seq.attention.model.types import LSTMState


class StackedLSTMEncoder(nn.Module):
    """Source embedding plus stacked LSTM. PAD must not update the final state.

    forward returns:
      encoder_outputs: (batch, src_len, hidden) top-layer hidden at each step
      state: final (h_n, c_n), each (num_layers, batch, hidden)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.padding_idx = padding_idx
        raise NotImplementedError("Implement StackedLSTMEncoder (Luong et al. 2015).")

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, LSTMState]:
        """src: (batch, src_len) token ids; src_lengths: (batch,)."""
        raise NotImplementedError
