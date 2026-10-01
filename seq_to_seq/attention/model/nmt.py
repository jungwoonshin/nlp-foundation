from __future__ import annotations

import torch
from torch import nn

from seq_to_seq.attention.config import LuongConfig
from seq_to_seq.attention.data import BOS_ID, EOS_ID, PAD_ID
from seq_to_seq.attention.model.global_attention import GlobalAttention
from seq_to_seq.attention.model.local_attention import LocalAttention
from seq_to_seq.attention.model.lstm import StackedLSTM
from seq_to_seq.attention.model.types import AttentionOutput, LSTMState


class LuongNMT(nn.Module):
    """Encoder-decoder NMT with global or local-m attention (Luong et al. 2015).

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
        *,
        max_source_length: int | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        if config.attention == "local_p":
            raise NotImplementedError(
                "local_p attention is not implemented."
            )
        if config.attention == "local_m" and config.score != "general":
            raise NotImplementedError(
                "local_m attention currently supports only general scoring."
            )
        embed_dim = config.embed_dim
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        max_source_length = (
            config.max_len if max_source_length is None else max_source_length
        )
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

        self.encoder = StackedLSTM(embed_dim, hidden_size, num_layers=num_layers)
        decoder_input_size = (
            embed_dim + hidden_size if config.input_feeding else embed_dim
        )
        self.decoder = StackedLSTM(decoder_input_size, hidden_size, num_layers=num_layers)
        if config.attention == "global":
            self.attention = GlobalAttention(
                hidden_size,
                score=config.score,
                max_source_length=max_source_length,
            )
        elif config.attention == "local_m":
            self.attention = LocalAttention(
                hidden_size=hidden_size,
                attention_kind=config.attention,
                window_size=config.window_size,
            )
        else:
            raise NotImplementedError(
                "LuongNMT only has Global or Local Attention."
            )
        # Equations (5) and (6) use matrix projections without bias terms.
        self.w_c = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.activation = nn.Tanh()
        self.out = nn.Linear(hidden_size, tgt_vocab_size, bias=False)

    def encode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, LSTMState]:
        embedded = self.encoder_embed(src)
        return self.encoder(embedded, lengths=src_lengths)

    def _decode_step(
        self,
        tokens: torch.Tensor,
        state: LSTMState,
        previous_attentional: torch.Tensor,
    ) -> tuple[torch.Tensor, LSTMState]:
        embedded = self.decoder_embed(tokens)
        if self.config.input_feeding:
            embedded = torch.cat([embedded, previous_attentional], dim=-1)
        outputs, state = self.decoder(embedded.unsqueeze(1), state=state)
        return outputs[:, 0], state

    def _attend(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_lengths: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, AttentionOutput]:
        if self.config.attention == "local_m":
            attention = self.attention(
                decoder_hidden, encoder_outputs, src_lengths, step=step
            )
        else:
            attention = self.attention(decoder_hidden, encoder_outputs, src_lengths)
        combined = torch.cat([attention.context, decoder_hidden], dim=-1)
        attentional_hidden = self.activation(self.w_c(combined))
        return attentional_hidden, attention

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_in: torch.Tensor,
    ) -> torch.Tensor:
        encoder_outputs, state = self.encode(src, src_lengths)
        previous_attentional = encoder_outputs.new_zeros(src.size(0), self.hidden_size)
        logits: list[torch.Tensor] = []
        for step in range(tgt_in.size(1)):
            decoder_hidden, state = self._decode_step(
                tgt_in[:, step], state, previous_attentional
            )
            previous_attentional, _ = self._attend(
                decoder_hidden, encoder_outputs, src_lengths, step
            )
            logits.append(self.out(previous_attentional))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        self.eval()
        batch = src.shape[0]
        encoder_outputs, state = self.encode(src, src_lengths)
        previous_attentional = encoder_outputs.new_zeros(batch, self.hidden_size)
        prev = src.new_full((batch,), BOS_ID)
        finished = torch.zeros(batch, dtype=torch.bool, device=src.device)
        steps: list[torch.Tensor] = []
        for step in range(max_len):
            decoder_hidden, state = self._decode_step(
                prev, state, previous_attentional
            )
            previous_attentional, _ = self._attend(
                decoder_hidden, encoder_outputs, src_lengths, step
            )
            next_token = self.out(previous_attentional).argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, PAD_ID), next_token
            )
            finished = finished | (next_token == EOS_ID)
            steps.append(next_token)
            prev = next_token
            if bool(finished.all()):
                break
        return torch.stack(steps, dim=1)
