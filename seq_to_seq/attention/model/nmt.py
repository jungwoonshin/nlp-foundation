from __future__ import annotations

import torch
from torch import nn

from seq_to_seq.attention.config import LuongConfig
from seq_to_seq.attention.model.types import LSTMState


class LuongNMT(nn.Module):
    """Encoder–decoder NMT with global or local attention (Luong et al. 2015).

    Training (teacher forcing):
      forward(src, src_lengths, tgt_in) -> logits (batch, tgt_len, tgt_vocab)

    Inference:
      generate(src, src_lengths, max_len) -> token ids (batch, decoded_len)
      greedy from BOS until EOS; finished rows filled with PAD.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        config: LuongConfig,
    ) -> None:
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.config = config
        raise NotImplementedError("Implement LuongNMT (Luong et al. 2015).")

    def encode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, LSTMState]:
        """Return encoder_outputs (batch, src_len, hidden) and final LSTM state."""
        raise NotImplementedError

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_in: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced logits: (batch, tgt_len, tgt_vocab)."""
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """Greedy decode, no BOS in the returned ids; EOS included if emitted."""
        raise NotImplementedError
