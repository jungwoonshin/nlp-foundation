from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from seq_to_seq.foundation.common.dataset import BOS_ID, EOS_ID, PAD_ID

Backend = Literal["scratch", "torch"]
LSTMState = tuple[torch.Tensor, torch.Tensor]


class LSTMCell(nn.Module):
    """LSTM cell with PyTorch gate order: input, forget, cell, output."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ih = nn.Linear(input_size, 4 * hidden_size)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        i_gate, f_gate, g_gate, o_gate = (self.ih(x_t) + self.hh(h_prev)).chunk(4, dim=-1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)
        c_t = f_gate * c_prev + i_gate * g_gate
        h_t = o_gate * torch.tanh(c_t)
        return h_t, c_t


class StackedLSTM(nn.Module):
    """Stack of LSTMCells. Layer 0 reads x; layer i reads layer i-1 at the same step."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [
                LSTMCell(input_size if layer == 0 else hidden_size, hidden_size)
                for layer in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        state: LSTMState | None = None,
    ) -> tuple[torch.Tensor, LSTMState]:
        batch, seq_len, _ = x.shape
        if state is None:
            h_layers = [x.new_zeros(batch, self.hidden_size) for _ in range(self.num_layers)]
            c_layers = [x.new_zeros(batch, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h_0, c_0 = state
            h_layers = [h_0[layer] for layer in range(self.num_layers)]
            c_layers = [c_0[layer] for layer in range(self.num_layers)]
        if lengths is not None:
            lengths = lengths.to(device=x.device)
        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            inp = x[:, t]
            mask = None if lengths is None else (t < lengths).unsqueeze(1)
            new_h: list[torch.Tensor] = []
            new_c: list[torch.Tensor] = []
            for layer, cell in enumerate(self.cells):
                h_t, c_t = cell(inp, h_layers[layer], c_layers[layer])
                if mask is None:
                    new_h.append(h_t)
                    new_c.append(c_t)
                    inp = h_t
                    continue
                h_keep = torch.where(mask, h_t, h_layers[layer])
                c_keep = torch.where(mask, c_t, c_layers[layer])
                new_h.append(h_keep)
                new_c.append(c_keep)
                inp = torch.where(mask, h_keep, torch.zeros_like(h_keep))
            h_layers = new_h
            c_layers = new_c
            if mask is None:
                outputs.append(h_layers[-1])
            else:
                outputs.append(torch.where(mask, h_layers[-1], torch.zeros_like(h_layers[-1])))
        h_n = torch.stack(h_layers, dim=0)
        c_n = torch.stack(c_layers, dim=0)
        return torch.stack(outputs, dim=1), (h_n, c_n)