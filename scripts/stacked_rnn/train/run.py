"""Train stacked-RNN seq2seq on reversed sequences."""

from __future__ import annotations

from seq_to_seq.foundation.common.training import run_reverse
from seq_to_seq.foundation.stacked_rnn.model import Seq2Seq

NUM_LAYERS = 2


def stacked_rnn(*, smoke: bool = False) -> None:
    def make_model(
        vocab_size: int,
        backend: str,
        embed_dim: int,
        hidden_size: int,
    ) -> Seq2Seq:
        return Seq2Seq(
            vocab_size,
            embed_dim,
            hidden_size,
            num_layers=NUM_LAYERS,
            backend=backend,  # type: ignore[arg-type]
        )

    run_reverse(make_model, smoke=smoke)


if __name__ == "__main__":
    stacked_rnn(smoke=True)
