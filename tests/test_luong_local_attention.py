from __future__ import annotations

import unittest
from dataclasses import replace

import torch
from torch.nn import functional as F

from seq_to_seq.attention.config import LuongConfig
from seq_to_seq.attention.model.local_attention import LocalAttention
from seq_to_seq.attention.model.nmt import LuongNMT


def reference_attention(
    decoder: torch.Tensor,
    encoder: torch.Tensor,
    weight: torch.Tensor,
    lengths: torch.Tensor,
    step: int,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute each allowed source score separately from the bilinear equation."""
    weights = encoder.new_zeros(encoder.shape[:2])
    contexts = []
    for b, length in enumerate(lengths.tolist()):
        positions = [s for s in range(length) if abs(s - step) <= radius]
        if not positions:
            # Keep zero gradients defined even when every window is empty.
            contexts.append(
                encoder[b].sum(dim=0) * 0 + decoder[b] * 0 + weight.sum() * 0
            )
            continue
        scores = torch.stack([decoder[b] @ weight @ encoder[b, s] for s in positions])
        probabilities = scores.softmax(dim=0)
        weights[b, positions] = probabilities
        contexts.append(
            sum(probabilities[i] * encoder[b, s] for i, s in enumerate(positions))
        )
    return weights, torch.stack(contexts)


class LocalAttentionTests(unittest.TestCase):
    def test_matches_equation_at_boundaries_with_padding_and_empty_windows(self) -> None:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for batch_size in (1, 2):
                for radius in (1, 2):
                    attention = LocalAttention(3, "local_m", window_size=radius).to(device)
                    weight = torch.tensor(
                        [[0.2, -0.3, 0.5], [0.7, 0.1, -0.2], [-0.4, 0.6, 0.8]],
                        device=device,
                    )
                    with torch.no_grad():
                        attention.global_attention.weight_a.weight.copy_(weight)
                    decoder = torch.tensor(
                        [[0.2, -0.4, 0.6], [-0.3, 0.1, 0.5]], device=device
                    )[:batch_size]
                    encoder = torch.arange(42, device=device, dtype=torch.float32)
                    encoder = encoder.reshape(2, 7, 3)[:batch_size] / 20
                    # CPU lengths are supported even when states are on CUDA.
                    lengths = torch.tensor([7, 3])[:batch_size]
                    observed_widths = []
                    handle = attention.global_attention.register_forward_hook(
                        lambda module, args, result: observed_widths.append(args[1].size(1))
                    )
                    try:
                        for step in (0, 3, 6, 7, 10):
                            with self.subTest(
                                device=device, batch=batch_size, radius=radius, step=step
                            ):
                                result = attention(decoder, encoder, lengths, step)
                                expected_weights, expected_context = reference_attention(
                                    decoder, encoder, weight, lengths, step, radius
                                )
                                torch.testing.assert_close(result.weights, expected_weights)
                                torch.testing.assert_close(result.context, expected_context)
                                self.assertEqual(result.p_t.tolist(), [step] * batch_size)
                                self.assertEqual(result.p_t.device, encoder.device)
                                self.assertEqual(result.weights.shape, (batch_size, 7))
                                self.assertEqual(result.context.shape, (batch_size, 3))
                                self.assertLessEqual(observed_widths[-1], 2 * radius + 1)
                                self.assertTrue(torch.isfinite(result.context).all())
                        self.assertEqual(len(observed_widths), 5)
                        self.assertEqual(observed_widths[-1], 0)
                    finally:
                        handle.remove()

    def test_gradients_match_reference_for_mixed_and_fully_empty_windows(self) -> None:
        for step in (4, 10):
            with self.subTest(step=step):
                torch.manual_seed(12)
                attention = LocalAttention(3, "local_m", window_size=1).double()
                decoder = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
                encoder = torch.randn(2, 7, 3, dtype=torch.double, requires_grad=True)
                lengths = torch.tensor([7, 3])
                ref_decoder = decoder.detach().clone().requires_grad_()
                ref_encoder = encoder.detach().clone().requires_grad_()
                weight = attention.global_attention.weight_a.weight
                ref_weight = weight.detach().clone().requires_grad_()

                actual = attention(decoder, encoder, lengths, step)
                ref_weights, ref_context = reference_attention(
                    ref_decoder, ref_encoder, ref_weight, lengths, step, 1
                )
                (actual.context.square().sum() + actual.weights.square().sum()).backward()
                (ref_context.square().sum() + ref_weights.square().sum()).backward()
                for value, reference in (
                    (decoder, ref_decoder), (encoder, ref_encoder), (weight, ref_weight)
                ):
                    self.assertIsNotNone(value.grad)
                    self.assertTrue(torch.isfinite(value.grad).all())
                    torch.testing.assert_close(value.grad, reference.grad)
                # The second example has no valid positions at either step.
                self.assertEqual(encoder.grad[1].count_nonzero().item(), 0)

    def test_rejects_unsupported_kind_and_invalid_window_inputs(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "Only local_m"):
            LocalAttention(3, "local_p")
        with self.assertRaisesRegex(ValueError, "window_size"):
            LocalAttention(3, "local_m", window_size=0)
        attention = LocalAttention(3, "local_m")
        decoder = torch.zeros(2, 3)
        encoder = torch.zeros(2, 5, 3)
        with self.assertRaisesRegex(ValueError, "step"):
            attention(decoder, encoder, torch.tensor([5, 3]), step=-1)
        for lengths in (torch.tensor([-1, 3]), torch.tensor([6, 3]), torch.tensor([[5, 3]])):
            with self.subTest(lengths=lengths):
                with self.assertRaisesRegex(ValueError, "src_lengths"):
                    attention(decoder, encoder, lengths, step=0)


class LocalAttentionNMTTests(unittest.TestCase):
    def test_forward_backward_optimizer_and_generation_receive_each_step(self) -> None:
        for batch_size, input_feeding in ((1, True), (2, True), (2, False)):
            with self.subTest(batch=batch_size, input_feeding=input_feeding):
                torch.manual_seed(21)
                config = replace(
                    LuongConfig.smoke(), attention="local_m", score="general",
                    window_size=1, hidden_size=4, embed_dim=4, max_len=2,
                    input_feeding=input_feeding,
                )
                model = LuongNMT(16, 16, config)
                self.assertEqual(model.attention.window_size, 1)
                source = torch.tensor([[4, 5, 6, 7, 0, 0], [8, 9, 10, 11, 12, 13]])[:batch_size]
                lengths = torch.tensor([4, 6])[:batch_size]
                target = torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10, 11]])
                target = target.expand(batch_size, -1)
                observed_steps = []
                handle = model.attention.register_forward_hook(
                    lambda module, args, result: observed_steps.append(result.p_t.tolist())
                )
                try:
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    weight = model.attention.global_attention.weight_a.weight
                    before = weight.detach().clone()
                    logits = model(source, lengths, target)
                    self.assertEqual(logits.shape, (batch_size, 9, 16))
                    self.assertTrue(torch.isfinite(logits).all())
                    self.assertEqual(observed_steps, [[t] * batch_size for t in range(9)])
                    loss = F.cross_entropy(logits.reshape(-1, 16), target.reshape(-1))
                    loss.backward()
                    self.assertTrue(torch.isfinite(weight.grad).all())
                    self.assertGreater(weight.grad.abs().sum().item(), 0)
                    optimizer.step()
                    self.assertFalse(torch.equal(weight, before))

                    observed_steps.clear()
                    # Force non-EOS predictions so generation passes the end of the source.
                    with torch.no_grad():
                        model.out.weight.zero_()
                    prediction = model.generate(source, lengths, max_len=9)
                    self.assertEqual(prediction.shape, (batch_size, 9))
                    self.assertEqual(observed_steps, [[t] * batch_size for t in range(9)])
                finally:
                    handle.remove()

    def test_rejects_unimplemented_local_combinations(self) -> None:
        for score in ("dot", "concat", "location"):
            with self.subTest(score=score):
                config = replace(LuongConfig.smoke(), attention="local_m", score=score)
                with self.assertRaisesRegex(NotImplementedError, "only general"):
                    LuongNMT(16, 16, config)
        config = replace(LuongConfig.smoke(), attention="local_p", score="general")
        with self.assertRaisesRegex(NotImplementedError, "local_p"):
            LuongNMT(16, 16, config)


if __name__ == "__main__":
    unittest.main()
