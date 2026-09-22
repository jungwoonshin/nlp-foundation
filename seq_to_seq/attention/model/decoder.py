from __future__ import annotations

import torch
from torch import nn

from seq_to_seq.attention.config import AlignmentScore, AttentionKind
from seq_to_seq.attention.model.types import AttentionOutput, LSTMState


class AttentionalDecoder(nn.Module):
    """Stacked LSTM decoder with Luong attention and optional input feeding.

    Input feeding (paper Fig. 4): concatenate h_tilde_{t-1} with the embedding
    of y_{t-1} as the first-layer LSTM input. When embed_dim == hidden_size,
    that first-layer input size is 2n.

    decode_step returns:
      logits: (batch, tgt_vocab)
      new_state: LSTMState
      h_tilde: (batch, hidden)
      attention: AttentionOutput
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        *,
        attention: AttentionKind,
        score: AlignmentScore,
        input_feeding: bool,
        dropout: float = 0.0,
        window_size: int = 10,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.score = score
        self.input_feeding = input_feeding
        self.dropout = dropout
        self.window_size = window_size
        self.padding_idx = padding_idx
        raise NotImplementedError("Implement AttentionalDecoder (Luong et al. 2015).")

    def decode_step(
        self,
        token: torch.Tensor,
        state: LSTMState,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
        prev_h_tilde: torch.Tensor | None,
        step: int,
    ) -> tuple[torch.Tensor, LSTMState, torch.Tensor, AttentionOutput]:
        """token: (batch,) previous target id (BOS on the first step)."""
        raise NotImplementedError
 