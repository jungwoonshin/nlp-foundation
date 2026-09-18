from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from seq_to_seq.foundation.common.dataset import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    ProcessedReverse,
    ReverseDataset,
)
from seq_to_seq.foundation.common.training import evaluate, fit
from seq_to_seq.foundation.simple_rnn.model import Seq2Seq as SimpleSeq2Seq
from seq_to_seq.foundation.simple_rnn.model import copy_into_torch as copy_simple
from seq_to_seq.foundation.stacked_lstm.model import Seq2Seq as StackedLSTMSeq2Seq
from seq_to_seq.foundation.stacked_lstm.model import copy_into_torch as copy_lstm
from seq_to_seq.foundation.stacked_rnn.model import Seq2Seq as StackedRNNSeq2Seq
from seq_to_seq.foundation.stacked_rnn.model import copy_into_torch as copy_stacked_rnn

VOCAB_SIZE = 12
EMBED_DIM = 8
HIDDEN_SIZE = 10
NUM_LAYERS = 2


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src = torch.tensor(
        [
            [3, 4, 5, PAD_ID],
            [6, 7, PAD_ID, PAD_ID],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([3, 2], dtype=torch.long)
    tgt_in = torch.tensor(
        [
            [BOS_ID, 5, 4, 3],
            [BOS_ID, 7, 6, PAD_ID],
        ],
        dtype=torch.long,
    )
    tgt_out = torch.tensor(
        [
            [5, 4, 3, EOS_ID],
            [7, 6, EOS_ID, PAD_ID],
        ],
        dtype=torch.long,
    )
    return src, lengths, tgt_in, tgt_out


class ReverseDatasetTests(unittest.TestCase):
    def test_target_is_reversed_with_bos_eos(self) -> None:
        item = ReverseDataset([[3, 4, 5]])[0]
        self.assertEqual(item["src"].tolist(), [3, 4, 5])
        self.assertEqual(item["tgt_in"].tolist(), [BOS_ID, 5, 4, 3])
        self.assertEqual(item["tgt_out"].tolist(), [5, 4, 3, EOS_ID])


class Seq2SeqContractMixin:
    make = staticmethod(lambda **kwargs: SimpleSeq2Seq(**kwargs))
    copy = staticmethod(copy_simple)
    is_lstm = False

    def _pair(self, backend: str, **extra):
        return self.make(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            hidden_size=HIDDEN_SIZE,
            backend=backend,
            **extra,
        )

    def _models(self, **extra):
        return self._pair("scratch", **extra), self._pair("torch", **extra)

    def test_forward_shapes_and_backward_both_backends(self) -> None:
        src, lengths, tgt_in, tgt_out = _batch()
        for backend in ("scratch", "torch"):
            with self.subTest(backend=backend):
                model = self._pair(backend)
                logits = model(src, lengths, tgt_in)
                self.assertEqual(tuple(logits.shape), (2, tgt_in.shape[1], VOCAB_SIZE))
                self.assertTrue(torch.isfinite(logits).all())
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    tgt_out.reshape(-1),
                    ignore_index=PAD_ID,
                )
                loss.backward()
                grads = [
                    param.grad
                    for name, param in model.named_parameters()
                    if param.requires_grad and "embed" not in name.lower()
                ]
                self.assertTrue(grads, "expected trainable non-embedding parameters")
                self.assertTrue(any(grad is not None for grad in grads))

    def test_pad_does_not_change_encoder_state(self) -> None:
        src, lengths, _, _ = _batch()
        src_short = torch.tensor([[3, 4, 5], [6, 7, PAD_ID]], dtype=torch.long)
        lengths_short = torch.tensor([3, 2], dtype=torch.long)
        for backend in ("scratch", "torch"):
            with self.subTest(backend=backend):
                model = self._pair(backend)
                model.eval()
                with torch.no_grad():
                    long_state = model.encode(src, lengths)
                    short_state = model.encode(src_short, lengths_short)
                self._assert_state_close(long_state, short_state)

    def test_scratch_matches_torch_after_weight_copy(self) -> None:
        src, lengths, tgt_in, _ = _batch()
        scratch, torch_model = self._models()
        self.copy(scratch, torch_model)
        scratch.eval()
        torch_model.eval()
        with torch.no_grad():
            self._assert_state_close(
                scratch.encode(src, lengths),
                torch_model.encode(src, lengths),
            )
            scratch_logits = scratch(src, lengths, tgt_in)
            torch_logits = torch_model(src, lengths, tgt_in)
        torch.testing.assert_close(scratch_logits, torch_logits, atol=1e-5, rtol=1e-5)

    def _assert_state_close(self, left, right) -> None:
        if self.is_lstm:
            torch.testing.assert_close(left[0], right[0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(left[1], right[1], atol=1e-5, rtol=1e-5)
            return
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)


class SimpleRNNTests(Seq2SeqContractMixin, unittest.TestCase):
    make = staticmethod(
        lambda **kwargs: SimpleSeq2Seq(
            vocab_size=kwargs["vocab_size"],
            embed_dim=kwargs["embed_dim"],
            hidden_size=kwargs["hidden_size"],
            backend=kwargs["backend"],
        )
    )
    copy = staticmethod(copy_simple)


class StackedRNNTests(Seq2SeqContractMixin, unittest.TestCase):
    make = staticmethod(
        lambda **kwargs: StackedRNNSeq2Seq(
            vocab_size=kwargs["vocab_size"],
            embed_dim=kwargs["embed_dim"],
            hidden_size=kwargs["hidden_size"],
            num_layers=kwargs.get("num_layers", NUM_LAYERS),
            backend=kwargs["backend"],
        )
    )
    copy = staticmethod(copy_stacked_rnn)


class StackedLSTMTests(Seq2SeqContractMixin, unittest.TestCase):
    make = staticmethod(
        lambda **kwargs: StackedLSTMSeq2Seq(
            vocab_size=kwargs["vocab_size"],
            embed_dim=kwargs["embed_dim"],
            hidden_size=kwargs["hidden_size"],
            num_layers=kwargs.get("num_layers", NUM_LAYERS),
            backend=kwargs["backend"],
        )
    )
    copy = staticmethod(copy_lstm)
    is_lstm = True


class SimpleRNNFitTests(unittest.TestCase):
    def test_scratch_overfits_tiny_reverse_set(self) -> None:
        sequences = [
            [3, 4],
            [4, 3],
            [3, 5],
            [5, 4],
            [4, 5, 3],
            [5, 3, 4],
        ]
        processed = ProcessedReverse(ReverseDataset(sequences), vocab_size=6)
        torch.manual_seed(0)
        model = SimpleSeq2Seq(vocab_size=6, embed_dim=16, hidden_size=32, backend="scratch")
        chosen = torch.device("cpu")
        fit(
            model,
            processed,
            chosen,
            epochs=40,
            batch_size=6,
            learning_rate=1e-2,
            verbose=False,
        )
        acc = evaluate(model, processed, chosen, batch_size=6)
        self.assertEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()
