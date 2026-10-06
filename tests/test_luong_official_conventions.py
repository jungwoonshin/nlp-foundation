from __future__ import annotations

import math
import unittest
from unittest import mock

import torch
from torch import nn

from seq_to_seq.attention.config import LuongConfig
from seq_to_seq.attention.data.dataset import LengthSortedBatchSampler, ParallelDataset, smoke_parallel
from seq_to_seq.attention.data.vocab import BOS_ID, EOS_ID, build_vocab
from seq_to_seq.attention.model.local_attention import LocalAttention
from seq_to_seq.attention.model.lstm import LSTMCell
from seq_to_seq.attention.model.nmt import LuongNMT
from seq_to_seq.attention.training.loop import initialize_parameters


class OfficialConventionsTests(unittest.TestCase):
    def test_lstm_has_no_bias_and_clips_state_after_computing_hidden(self):
        cell = LSTMCell(1, 1).double()
        self.assertIsNone(cell.ih.bias)
        self.assertIsNone(cell.hh.bias)
        with torch.no_grad():
            cell.ih.weight.zero_()
            cell.hh.weight.zero_()
        hidden, state = cell(torch.zeros(2, 1).double(), torch.zeros(2, 1).double(),
                             torch.tensor([[200.0], [-200.0]]).double())
        torch.testing.assert_close(state, torch.tensor([[50.0], [-50.0]]).double())
        torch.testing.assert_close(hidden, torch.tensor([[0.5], [-0.5]]).double())

    def test_target_specials_preserve_content_ids_and_match_official_size(self):
        base = build_vocab([str(i) for i in range(997)])
        target = base.for_target()
        self.assertEqual(target.size, 1002)
        self.assertEqual(target.id_to_token[BOS_ID], "<t_sos>")
        self.assertEqual(target.id_to_token[EOS_ID], "<t_eos>")
        self.assertEqual(target.token_to_id["<s>"], 0)
        self.assertNotEqual(target.token_to_id["</s>"], EOS_ID)
        self.assertEqual(target.encode(["10", "unknown"]), base.encode(["10", "unknown"]))
        self.assertIs(target.for_target(), target)
        processed = smoke_parallel()
        model = LuongNMT(processed.src_vocab_size, processed.tgt_vocab_size, LuongConfig.smoke())
        self.assertEqual(model.out.out_features, processed.tgt_vocab.size)

    def test_sorted_chunks_shuffle_batches_without_losing_or_crossing_examples(self):
        dataset = ParallelDataset([[i + 4] for i in range(211)],
                                  [[4] * ((211 - i) % 17 + 1) for i in range(211)])
        torch.manual_seed(7)
        batches = list(LengthSortedBatchSampler(dataset, 2))
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(211)))
        self.assertEqual(len(batches), 106)
        # The first 100 batches stay inside the first chunk of 200 examples.
        self.assertTrue(all(i < 200 for batch in batches[:100] for i in batch))
        self.assertTrue(all(i >= 200 for batch in batches[100:] for i in batch))
        sorted_indices = sorted(range(200), key=lambda i: len(dataset.targets[i]))
        expected = {tuple(sorted_indices[i:i + 2]) for i in range(0, 200, 2)}
        self.assertEqual({tuple(b) for b in batches[:100]}, expected)
        torch.manual_seed(7)
        self.assertEqual(batches, list(LengthSortedBatchSampler(dataset, 2)))

    def test_explicit_seed_is_reproducible_and_zero_uses_clock(self):
        model = nn.Linear(2, 3)
        initialize_parameters(model, seed=7)
        before = model.weight.detach().clone()
        initialize_parameters(model, seed=7)
        torch.testing.assert_close(model.weight, before)
        with mock.patch("seq_to_seq.attention.training.loop.time.time_ns", return_value=123), \
             mock.patch("torch.manual_seed") as seed:
            initialize_parameters(model, seed=0)
        seed.assert_called_once_with(123)

    def test_local_windows_match_one_based_official_coordinates(self):
        for kind in ("local_m", "local_p"):
            for reverse in (False, True):
                with self.subTest(kind=kind, reverse=reverse):
                    attention = LocalAttention(2, kind, window_size=1, reverse_source=reverse).double()
                    decoder = torch.tensor([[0.2, -0.3]] * 3).double()
                    encoder = torch.arange(36).reshape(3, 6, 2).double() / 20
                    lengths = torch.tensor([6, 3, 0])
                    # A fixed fractional prediction tests the 3-slot floor-anchored window.
                    centers = torch.tensor([2.5, 0.75, 0.0]).double()
                    with mock.patch.object(attention, "predictive_p_t", return_value=centers):
                        actual = attention(decoder, encoder, lengths, step=20)
                    expected = torch.zeros(3, 6).double()
                    weight = (attention.gaussian_attention if kind == "local_p"
                              else attention.global_attention).weight_a.weight
                    for b, length in enumerate(lengths.tolist()):
                        # Reference uses the MATLAB code's one-based mu and positions.
                        mu = centers[b].item() + 1 if kind == "local_p" else min(21, length)
                        positions = [s for s in range(1, length + 1) if abs(s - math.floor(mu)) <= 1]
                        if not positions:
                            continue
                        indices = [length - s if reverse else s - 1 for s in positions]
                        scores = torch.stack([decoder[b] @ weight @ encoder[b, i] for i in indices])
                        probabilities = scores.softmax(0)
                        if kind == "local_p":
                            probabilities = probabilities * torch.exp(
                                -(torch.tensor(positions).double() - mu).square() / 0.5
                            )
                        expected[b, indices] = probabilities
                    torch.testing.assert_close(actual.weights, expected)
                    torch.testing.assert_close(actual.context, torch.einsum("bs,bsh->bh", expected, encoder))


if __name__ == "__main__":
    unittest.main()
