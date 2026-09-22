from __future__ import annotations

import torch
from torch import nn

from seq_to_seq.attention.data.vocab import BOS_ID, EOS_ID, PAD_ID


class DummyNMT(nn.Module):
    """Embedding + linear stand-in so data/train/eval can run without LuongNMT."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if src_vocab_size <= EOS_ID or tgt_vocab_size <= EOS_ID:
            raise ValueError("vocab sizes must include specials")
        if embed_dim < 1 or hidden_size < 1:
            raise ValueError("embed_dim and hidden_size must be >= 1")
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=PAD_ID)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=PAD_ID)
        self.out = nn.Linear(embed_dim * 2, tgt_vocab_size)

    def _source_vec(self, src: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.src_embed(src)
        lengths = src_lengths.to(device=src.device, dtype=embedded.dtype).clamp(min=1)
        mask = (src != PAD_ID).unsqueeze(-1).to(dtype=embedded.dtype)
        summed = (embedded * mask).sum(dim=1)
        return summed / lengths.unsqueeze(1)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_in: torch.Tensor,
    ) -> torch.Tensor:
        src_vec = self._source_vec(src, src_lengths)
        tgt_e = self.tgt_embed(tgt_in)
        src_exp = src_vec.unsqueeze(1).expand(-1, tgt_in.shape[1], -1)
        return self.out(torch.cat([src_exp, tgt_e], dim=-1))

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        self.eval()
        if max_len < 1:
            raise ValueError("max_len must be >= 1")
        batch = src.shape[0]
        prev = src.new_full((batch, 1), BOS_ID)
        finished = torch.zeros(batch, dtype=torch.bool, device=src.device)
        steps: list[torch.Tensor] = []
        for _ in range(max_len):
            logits = self.forward(src, src_lengths, prev)
            next_token = logits[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, PAD_ID), next_token)
            finished = finished | (next_token == EOS_ID)
            steps.append(next_token)
            prev = next_token.unsqueeze(1)
            if bool(finished.all()):
                break
        return torch.stack(steps, dim=1)
