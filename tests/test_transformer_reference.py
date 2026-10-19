"""Regression checks for the pinned Multi30k reference conventions."""
import math
import unittest
import torch
from transformer.model.positional_encoding import SinusoidalPositionalEncoding
from transformer.training.adam import ReferenceAdam
from transformer.model.attention import MultiHeadAttention


class ReferenceTests(unittest.TestCase):
    def test_positions_frequency_offset_padding_and_odd_dimension(self):
        for d in (7, 8):
            m = SinusoidalPositionalEncoding(d, 4, 0.)
            mask = torch.tensor([[True, True, False, False], [False, False, True, True]])
            result = m(torch.zeros(2, 4, d), mask)
            torch.testing.assert_close(result[0, 2:], result[1, :2])
            self.assertEqual(result[0, :2].count_nonzero(), 0)
            self.assertAlmostEqual(float(result[0, 2, 0]), math.sin(2), places=6)
            self.assertAlmostEqual(float(result[0, 2, d//2]), math.cos(2), places=6)
            self.assertAlmostEqual(float(result[0, 2, d//2-1]), math.sin(2/10000), places=6)
            self.assertTrue(torch.isfinite(m(torch.zeros(1, 4, d), torch.zeros(1, 4, dtype=torch.bool))).all())
            if d % 2:
                self.assertEqual(result[..., -1].count_nonzero(), 0)

    def test_adam_epsilon_before_bias_correction(self):
        p = torch.nn.Parameter(torch.tensor([.2], dtype=torch.double))
        opt = ReferenceAdam([p], lr=.1, betas=(.9, .98), eps=.01)
        p.grad = torch.tensor([.3], dtype=torch.double)
        opt.step()
        expected = .2 - (.1 * math.sqrt(.02) / .1) * (.1 * .3) / (math.sqrt(.02 * .3**2) + .01)
        self.assertAlmostEqual(p.item(), expected, places=14)

    def test_attention_biases_and_blocked_rows(self):
        m = MultiHeadAttention(8, 2)
        self.assertIsNotNone(m.q_proj.bias)
        self.assertEqual(m.o_proj.bias.count_nonzero(), 0)
        x = torch.randn(2, 3, 8, requires_grad=True)
        out = m(x, x, x, torch.ones(2, 1, 3, 3, dtype=torch.bool))
        self.assertTrue(torch.isfinite(out.context).all())
        self.assertEqual(out.weights.count_nonzero(), 0)
        out.context.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
