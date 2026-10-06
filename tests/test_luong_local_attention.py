from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

import torch
from torch.nn import functional as F

from seq_to_seq.attention.config import LuongConfig
from seq_to_seq.attention.model.global_attention import GlobalAttention
from seq_to_seq.attention.model.local_attention import LocalAttention
from seq_to_seq.attention.model.nmt import LuongNMT
from seq_to_seq.attention.model.types import AttentionOutput


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
        center = min(step, max(length - 1, 0))
        positions = [s for s in range(length) if abs(s - center) <= radius]
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
                                self.assertEqual(result.p_t.tolist(), [min(step, n - 1) for n in lengths.tolist()])
                                self.assertEqual(result.p_t.device, encoder.device)
                                self.assertEqual(result.weights.shape, (batch_size, 7))
                                self.assertEqual(result.context.shape, (batch_size, 3))
                                self.assertLessEqual(observed_widths[-1], 2 * radius + 1)
                                self.assertTrue(torch.isfinite(result.context).all())
                        self.assertEqual(len(observed_widths), 5)
                        self.assertEqual(observed_widths[-1], min(7, 2 * radius + 1))
                    finally:
                        handle.remove()

    def test_gradients_match_reference_for_clamped_and_empty_sources(self) -> None:
        for step in (4, 10):
            with self.subTest(step=step):
                torch.manual_seed(12)
                attention = LocalAttention(3, "local_m", window_size=1).double()
                decoder = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
                encoder = torch.randn(2, 7, 3, dtype=torch.double, requires_grad=True)
                lengths = torch.tensor([7, 0])
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
                # The empty source contributes no encoder gradient.
                self.assertEqual(encoder.grad[1].count_nonzero().item(), 0)

    def test_rejects_unsupported_kind_and_invalid_window_inputs(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "Only local_m and local_p"):
            LocalAttention(3, "global")
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
                    expected_steps = [[min(t, n - 1) for n in lengths.tolist()] for t in range(9)]
                    self.assertEqual(observed_steps, expected_steps)
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
                    self.assertEqual(observed_steps, expected_steps)
                finally:
                    handle.remove()

    def test_local_p_construction_and_step_dispatch(self) -> None:
        config = replace(
            LuongConfig.smoke(), attention="local_p", score="general",
            hidden_size=3, embed_dim=3, window_size=2,
        )
        model = LuongNMT(16, 16, config)
        attention = model.attention
        self.assertIsInstance(attention, LocalAttention)
        self.assertEqual(attention.attention_kind, "local_p")
        self.assertEqual(attention.weight_p.weight.shape, (3, 3))
        self.assertEqual(attention.weight_va.weight.shape, (1, 3))
        self.assertTrue(attention.gaussian_attention.is_gaussian)
        self.assertEqual(attention.gaussian_attention.sigma, 1.0)

        decoder = torch.zeros(2, 3)
        encoder = torch.zeros(2, 5, 3)
        lengths = torch.tensor([5, 3])
        output = AttentionOutput(
            context=torch.zeros(2, 3), weights=torch.zeros(2, 5),
            p_t=torch.tensor([2.5, 1.5]),
        )
        # Isolate the NMT routing contract from attention arithmetic.
        with mock.patch.object(attention, "forward", return_value=output) as forward:
            attentional, result = model._attend(decoder, encoder, lengths, step=4)
        forward.assert_called_once_with(decoder, encoder, lengths, step=4)
        self.assertIs(result, output)
        self.assertEqual(attentional.shape, (2, 3))

    def test_rejects_unimplemented_local_combinations(self) -> None:
        for kind in ("local_m", "local_p"):
            for score in ("dot", "concat", "location"):
                with self.subTest(kind=kind, score=score):
                    config = replace(LuongConfig.smoke(), attention=kind, score=score)
                    with self.assertRaisesRegex(NotImplementedError, "only general"):
                        LuongNMT(16, 16, config)


class PredictiveAttentionTests(unittest.TestCase):
    def test_local_p_window_does_not_round_a_fractional_center_to_an_integer(self) -> None:
        attention = LocalAttention(2, "local_p", window_size=10)
        center = torch.nextafter(torch.tensor([2.0]), torch.tensor([0.0]))
        with mock.patch.object(attention, "predictive_p_t", return_value=center):
            result = attention(torch.zeros(1, 2), torch.ones(1, 20, 2), torch.tensor([20]), step=0)
        self.assertGreater(result.weights[0, 11].item(), 0)
        self.assertEqual(result.weights[0, 12].item(), 0)

    def test_batched_local_p_matches_separate_examples_and_gradients(self) -> None:
        for source_len in (0, 1, 7):
            with self.subTest(source_len=source_len):
                torch.manual_seed(41)
                attention = LocalAttention(3, "local_p", window_size=2).double()
                reference = LocalAttention(3, "local_p", window_size=2).double()
                reference.load_state_dict(attention.state_dict())
                decoder = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
                encoder = torch.randn(4, source_len, 3, dtype=torch.double, requires_grad=True)
                lengths = torch.tensor([source_len, min(3, source_len), min(1, source_len), 0])
                ref_decoder = decoder.detach().clone().requires_grad_()
                ref_encoder = encoder.detach().clone().requires_grad_()
                actual = attention(decoder, encoder, lengths, step=0)
                centers = reference.predictive_p_t(ref_decoder, lengths)
                contexts, weights = [], []
                for b, center in enumerate(centers):
                    positions = [s for s in range(int(lengths[b])) if abs(s - int(center.item())) <= 2]
                    start = positions[0] if positions else 0
                    end = positions[-1] + 1 if positions else 0
                    result = reference.gaussian_attention(
                        ref_decoder[b:b + 1], ref_encoder[b:b + 1, start:end],
                        lengths.new_tensor([end - start]), center.reshape(1), source_start=start,
                    )
                    contexts.append(result.context)
                    weights.append(F.pad(result.weights, (start, source_len - end)))
                ref_context, ref_weights = torch.cat(contexts), torch.cat(weights)
                torch.testing.assert_close(actual.context, ref_context)
                torch.testing.assert_close(actual.weights, ref_weights)
                torch.testing.assert_close(actual.p_t, centers)
                (actual.context.square().sum() + actual.weights.square().sum()).backward()
                (ref_context.square().sum() + ref_weights.square().sum()).backward()
                for value, expected in [(decoder, ref_decoder), (encoder, ref_encoder)]:
                    torch.testing.assert_close(value.grad, expected.grad)
                for name, parameter in attention.named_parameters():
                    torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad)

    def test_gaussian_matches_equation_with_batched_centers_and_source_offset(self) -> None:
        attention = GlobalAttention(2, "general", 5, is_gaussian=True).double()
        with torch.no_grad():
            attention.weight_a.weight.copy_(torch.tensor([[0.2, -0.3], [0.7, 0.1]]))
        decoder = torch.tensor([[0.3, -0.4], [-0.2, 0.5]], dtype=torch.double)
        encoder = torch.arange(16, dtype=torch.double).reshape(2, 4, 2) / 10
        centers = torch.tensor([0.0, 3.5], dtype=torch.double, requires_grad=True)
        lengths = torch.tensor([4, 2])
        actual = attention(decoder, encoder, lengths, centers, source_start=2)
        expected_rows = []
        for b, length in enumerate(lengths.tolist()):
            scores = torch.stack([
                decoder[b] @ attention.weight_a.weight @ encoder[b, s]
                for s in range(length)
            ])
            distances = torch.arange(2, 2 + length, dtype=torch.double) - centers[b]
            gaussian = torch.exp(-distances.square() / 2)
            expected_rows.append(F.pad(scores.softmax(0) * gaussian, (0, 4 - length)))
        expected = torch.stack(expected_rows)
        torch.testing.assert_close(actual.weights, expected)
        torch.testing.assert_close(actual.context, torch.einsum("bs,bsh->bh", expected, encoder))
        self.assertTrue((actual.weights.sum(-1) < 1).all())
        actual_gradient = torch.autograd.grad(actual.context.sum(), centers, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad((expected.unsqueeze(-1) * encoder).sum(), centers)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertTrue((actual_gradient.abs() > 0).all())

    def test_gaussian_handles_empty_sources_and_requires_centers(self) -> None:
        attention = GlobalAttention(2, "general", 3, is_gaussian=True)
        decoder, encoder = torch.zeros(2, 2), torch.zeros(2, 0, 2)
        result = attention(decoder, encoder, torch.zeros(2, dtype=torch.long), torch.zeros(2))
        self.assertEqual(result.weights.shape, (2, 0))
        torch.testing.assert_close(result.context, torch.zeros(2, 2))
        with self.assertRaisesRegex(ValueError, "requires p_t"):
            attention(decoder, encoder, torch.zeros(2, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "shape"):
            attention(decoder, encoder, torch.zeros(2, dtype=torch.long), torch.zeros(2, 1))
        with self.assertRaisesRegex(ValueError, "max_source_length"):
            GlobalAttention(2, "general", 1, is_gaussian=True)

    def test_local_p_windows_match_reference_and_preserve_centers_and_hooks(self) -> None:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for batch in (1, 3):
                with self.subTest(device=device, batch=batch):
                    attention = LocalAttention(2, "local_p", window_size=1).to(device)
                    with torch.no_grad():
                        attention.weight_p.weight.zero_()
                        attention.weight_va.weight.fill_(1)
                    decoder = torch.tensor([[0.3, -0.4]] * batch, device=device, requires_grad=True)
                    encoder = torch.arange(batch * 14, device=device, dtype=torch.float32).reshape(batch, 7, 2) / 10
                    encoder.requires_grad_()
                    lengths = torch.tensor([7, 3, 0])[:batch]
                    calls = []
                    handle = attention.gaussian_attention.register_forward_hook(
                        lambda module, args, output: calls.append(args[1].size(1))
                    )
                    try:
                        actual = attention(decoder, encoder, lengths, step=0)
                    finally:
                        handle.remove()
                    torch.testing.assert_close(actual.p_t, lengths.to(device) * 0.5)
                    self.assertTrue(actual.p_t.is_floating_point())
                    self.assertTrue(actual.p_t.requires_grad)
                    self.assertEqual(len(calls), 1)
                    self.assertLessEqual(calls[0], 3)
                    expected = torch.zeros(batch, 7, device=device)
                    for b, length in enumerate(lengths.tolist()):
                        center = length * 0.5
                        positions = [s for s in range(length) if abs(s - int(center)) <= 1]
                        if positions:
                            scores = torch.stack([
                                decoder[b] @ attention.gaussian_attention.weight_a.weight @ encoder[b, s]
                                for s in positions
                            ])
                            distances = torch.tensor(positions, device=device) - center
                            expected[b, positions] = scores.softmax(0) * torch.exp(-distances.square() / 0.5)
                    torch.testing.assert_close(actual.weights, expected)
                    torch.testing.assert_close(actual.context, torch.einsum("bs,bsh->bh", expected, encoder))
                    actual.context.square().sum().backward()
                    self.assertTrue(torch.isfinite(attention.weight_p.weight.grad).all())
                    if batch == 3:
                        self.assertEqual(encoder.grad[2].count_nonzero().item(), 0)

    def test_local_p_nmt_trains_and_generates_with_predictor_gradients(self) -> None:
        for batch, input_feeding in ((1, True), (2, True), (2, False)):
            with self.subTest(batch=batch, input_feeding=input_feeding):
                torch.manual_seed(25)
                config = replace(
                    LuongConfig.smoke(), attention="local_p", score="general",
                    hidden_size=4, embed_dim=4, window_size=2, input_feeding=input_feeding,
                )
                model = LuongNMT(16, 16, config)
                source = torch.tensor([[4, 5, 6, 7, 0, 0], [8, 9, 10, 11, 12, 13]])[:batch]
                lengths = torch.tensor([4, 6])[:batch]
                target = torch.tensor([[1, 4, 5, 6, 7]]).expand(batch, -1)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                predictor = model.attention.weight_va.weight
                before = predictor.detach().clone()
                logits = model(source, lengths, target)
                self.assertEqual(logits.shape, (batch, 5, 16))
                F.cross_entropy(logits.reshape(-1, 16), target.reshape(-1)).backward()
                for parameter in (
                    model.attention.weight_p.weight, predictor,
                    model.attention.gaussian_attention.weight_a.weight,
                ):
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                optimizer.step()
                self.assertFalse(torch.equal(predictor, before))
                with torch.no_grad():
                    model.out.weight.zero_()
                self.assertEqual(model.generate(source, lengths, max_len=5).shape, (batch, 5))


if __name__ == "__main__":
    unittest.main()
