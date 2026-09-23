from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from seq_to_seq.attention.config import LuongConfig
from seq_to_seq.attention.data import EOS_ID, PAD_ID
from seq_to_seq.attention.model.types import LSTMState
from seq_to_seq.attention.model.lstm import StackedLSTM


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
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        config: LuongConfig,
    ) -> None:
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.config = config

        if src_vocab_size <= EOS_ID:
            raise ValueError("src_vocab_size must include PAD, BOS, EOS, and content tokens")
        if tgt_vocab_size <= EOS_ID:
            raise ValueError("tgt_vocab_size must include PAD, BOS, EOS, and content tokens")
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=PAD_ID)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=PAD_ID)

        self.encoder = StackedLSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = StackedLSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, tgt_vocab_size)

    def encode(self, src: torch.Tensor, src_lengths: torch.Tensor) -> LSTMState:
        embedded = self.encoder_embed(src)
        packed = pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.encoder(packed)
        return state

    def _decode(
        self,
        tokens: torch.Tensor,
        state: LSTMState,
    ) -> tuple[torch.Tensor, LSTMState]:
        embedded = self.decoder_embed(tokens)
        return self.decoder(embedded, state)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_in: torch.Tensor,
    ) -> torch.Tensor:
        state = self.encode(src, src_lengths)
        outputs, _ = self._decode(tgt_in, state)
        return self.out(outputs)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        self.eval()
        batch = src.shape[0]
        state = self.encode(src, src_lengths)
        prev = src.new_full((batch, 1), BOS_ID)
        finished = torch.zeros(batch, dtype=torch.bool, device=src.device)
        steps: list[torch.Tensor] = []
        for _ in range(max_len):
            outputs, state = self._decode(prev, state)
            next_token = self.out(outputs[:, -1]).argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, PAD_ID), next_token)
            finished = finished | (next_token == EOS_ID)
            steps.append(next_token)
            prev = next_token.unsqueeze(1)
            if bool(finished.all()):
                break
        return torch.stack(steps, dim=1)