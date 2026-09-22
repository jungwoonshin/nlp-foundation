from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from seq_to_seq.foundation.common.dataset import BOS_ID, EOS_ID, PAD_ID

Backend = Literal["scratch", "torch"]


class RNNCell(nn.Module):
    """Vanilla Elman cell: h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.ih = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.ih(x_t) + self.hh(h_prev))


class SimpleRNN(nn.Module):
    """Unroll one RNNCell over time. `h_0` is (1, batch, hidden) like nn.RNN."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.cell = RNNCell(input_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        h_0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        h = x.new_zeros(batch, self.hidden_size) if h_0 is None else h_0[0]
        if lengths is not None:
            lengths = lengths.to(device=x.device)
        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            h_t = self.cell(x[:, t], h)
            if lengths is None:
                h = h_t
                outputs.append(h)
                continue
            mask = (t < lengths).unsqueeze(1)
            h = torch.where(mask, h_t, h)
            outputs.append(torch.where(mask, h, torch.zeros_like(h)))
        
        # torch.stack(outputs, dim=1) -> (batch, seq_len, hidden_size)
        # h.unsqueeze(0) -> (1, batch, hidden_size)
        return torch.stack(outputs, dim=1), h.unsqueeze(0) 


class Seq2Seq(nn.Module):
    """1-layer encoder-decoder. `backend` is 'scratch' (RNNCell) or 'torch' (nn.RNN)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        *,
        backend: Backend = "scratch",
    ) -> None:
        super().__init__()
        if vocab_size <= EOS_ID:
            raise ValueError("vocab_size must include PAD, BOS, EOS, and content tokens")
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if backend not in ("scratch", "torch"):
            raise ValueError("backend must be 'scratch' or 'torch'")
        self.backend = backend
        self.hidden_size = hidden_size
        self.encoder_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.decoder_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        if backend == "scratch":
            self.encoder: SimpleRNN | nn.RNN = SimpleRNN(embed_dim, hidden_size)
            self.decoder: SimpleRNN | nn.RNN = SimpleRNN(embed_dim, hidden_size)
        else:
            self.encoder = nn.RNN(embed_dim, hidden_size, batch_first=True)
            self.decoder = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def encode(self, src: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.encoder_embed(src)
        if self.backend == "torch":
            packed = pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, state = self.encoder(packed)
            return state
        _, state = self.encoder(embedded, lengths=src_lengths)
        return state

    def _decode(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.decoder_embed(tokens)
        if self.backend == "scratch":
            # embedded is (batch, seq_len, embed_dim) -> [BOS, Token0, Token1, ...] -> Input X for RNN Cell
            # state is (1, batch, hidden_size) -> Hidden State from Encode Step
            # return is (batch, seq_len, hidden_size)
            return self.decoder(embedded, h_0=state) # Teacher Forcing : Input X is the target token
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


def copy_rnn_cell_into_torch(cell: RNNCell, rnn: nn.RNN, layer: int = 0) -> None:
    getattr(rnn, f"weight_ih_l{layer}").data.copy_(cell.ih.weight.data)
    getattr(rnn, f"weight_hh_l{layer}").data.copy_(cell.hh.weight.data)
    getattr(rnn, f"bias_ih_l{layer}").data.copy_(cell.ih.bias.data)
    getattr(rnn, f"bias_hh_l{layer}").data.copy_(cell.hh.bias.data)


def copy_into_torch(scratch: Seq2Seq, torch_model: Seq2Seq) -> None:
    if scratch.backend != "scratch" or torch_model.backend != "torch":
        raise ValueError("copy_into_torch expects scratch -> torch")
    torch_model.encoder_embed.weight.data.copy_(scratch.encoder_embed.weight.data)
    torch_model.decoder_embed.weight.data.copy_(scratch.decoder_embed.weight.data)
    torch_model.out.weight.data.copy_(scratch.out.weight.data)
    torch_model.out.bias.data.copy_(scratch.out.bias.data)
    assert isinstance(scratch.encoder, SimpleRNN)
    assert isinstance(scratch.decoder, SimpleRNN)
    assert isinstance(torch_model.encoder, nn.RNN)
    assert isinstance(torch_model.decoder, nn.RNN)
    copy_rnn_cell_into_torch(scratch.encoder.cell, torch_model.encoder)
    copy_rnn_cell_into_torch(scratch.decoder.cell, torch_model.decoder)
