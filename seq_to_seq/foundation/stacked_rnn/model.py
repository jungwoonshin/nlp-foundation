from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from seq_to_seq.foundation.common.dataset import BOS_ID, EOS_ID, PAD_ID
from seq_to_seq.foundation.simple_rnn.model import RNNCell, copy_rnn_cell_into_torch

Backend = Literal["scratch", "torch"]


class StackedRNN(nn.Module):
    """Stack of RNNCells. Layer 0 reads x; layer i reads layer i-1 at the same step."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [
                RNNCell(input_size if layer == 0 else hidden_size, hidden_size)
                for layer in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        h_0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        if h_0 is None:
            h_layers = [x.new_zeros(batch, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h_layers = [h_0[layer] for layer in range(self.num_layers)]
        if lengths is not None:
            lengths = lengths.to(device=x.device)
        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            inp = x[:, t]
            mask = None if lengths is None else (t < lengths).unsqueeze(1)
            new_h: list[torch.Tensor] = []
            for layer, cell in enumerate(self.cells):
                h_t = cell(inp, h_layers[layer])
                if mask is None:
                    new_h.append(h_t)
                    inp = h_t
                    continue
                h_keep = torch.where(mask, h_t, h_layers[layer])
                new_h.append(h_keep)
                inp = torch.where(mask, h_keep, torch.zeros_like(h_keep))
            h_layers = new_h
            if mask is None:
                outputs.append(h_layers[-1])
            else:
                outputs.append(torch.where(mask, h_layers[-1], torch.zeros_like(h_layers[-1])))
        return torch.stack(outputs, dim=1), torch.stack(h_layers, dim=0)


class Seq2Seq(nn.Module):
    """Stacked-RNN encoder-decoder. `backend` is 'scratch' or 'torch' (nn.RNN)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 2,
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
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if backend not in ("scratch", "torch"):
            raise ValueError("backend must be 'scratch' or 'torch'")
        self.backend = backend
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.decoder_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        if backend == "scratch":
            self.encoder: StackedRNN | nn.RNN = StackedRNN(embed_dim, hidden_size, num_layers)
            self.decoder: StackedRNN | nn.RNN = StackedRNN(embed_dim, hidden_size, num_layers)
        else:
            self.encoder = nn.RNN(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
            self.decoder = nn.RNN(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
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
            return self.decoder(embedded, h_0=state)
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


def copy_into_torch(scratch: Seq2Seq, torch_model: Seq2Seq) -> None:
    if scratch.backend != "scratch" or torch_model.backend != "torch":
        raise ValueError("copy_into_torch expects scratch -> torch")
    if scratch.num_layers != torch_model.num_layers:
        raise ValueError("num_layers must match")
    torch_model.encoder_embed.weight.data.copy_(scratch.encoder_embed.weight.data)
    torch_model.decoder_embed.weight.data.copy_(scratch.decoder_embed.weight.data)
    torch_model.out.weight.data.copy_(scratch.out.weight.data)
    torch_model.out.bias.data.copy_(scratch.out.bias.data)
    assert isinstance(scratch.encoder, StackedRNN)
    assert isinstance(scratch.decoder, StackedRNN)
    assert isinstance(torch_model.encoder, nn.RNN)
    assert isinstance(torch_model.decoder, nn.RNN)
    for layer, cell in enumerate(scratch.encoder.cells):
        copy_rnn_cell_into_torch(cell, torch_model.encoder, layer)
    for layer, cell in enumerate(scratch.decoder.cells):
        copy_rnn_cell_into_torch(cell, torch_model.decoder, layer)
