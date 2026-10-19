from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from lstm_attention.attention.config import LuongConfig
from lstm_attention.attention.training.loop import _clip_gradients, fit


class LuongTrainingTests(unittest.TestCase):
    def test_sgd_update_matches_summed_sentence_loss_with_padding(self) -> None:
        class LogitsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(2, 3, 5, dtype=torch.double))

            def forward(self, src, lengths, tgt):
                return self.logits

        targets = torch.tensor([[3, 4, 2], [4, 2, 0]])
        batch = dict(src=torch.ones(2, 2, dtype=torch.long), src_lengths=torch.tensor([2, 2]),
                     tgt_in=targets, tgt_out=targets)
        processed = SimpleNamespace(dataloader=lambda *args, **kwargs: [batch])
        model = LogitsModel()
        expected = model.logits.detach().clone().requires_grad_()
        # Sum each sentence's valid token losses, then average the two sentences.
        loss = sum(F.cross_entropy(expected[b, targets[b] != 0], targets[b, targets[b] != 0],
                                   reduction="sum") for b in range(2)) / 2
        loss.backward()
        config = replace(LuongConfig.smoke(), epochs=1, optimizer="sgd",
                         learning_rate=0.1, grad_clip=100.0)
        metrics = fit(model, processed, torch.device("cpu"), config, verbose=False)
        torch.testing.assert_close(model.logits, expected.detach() - 0.1 * expected.grad)
        self.assertAlmostEqual(metrics["loss"], loss.item())
        self.assertEqual(model.logits.grad[1, 2].count_nonzero().item(), 0)

    def test_short_batch_uses_configured_denominator_but_reports_per_sentence_loss(self) -> None:
        class LogitsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(1, 2, 5, dtype=torch.double))

            def forward(self, src, lengths, tgt):
                return self.logits

        targets = torch.tensor([[4, 2]])
        batch = dict(src=torch.tensor([[4]]), src_lengths=torch.tensor([1]),
                     tgt_in=torch.tensor([[1, 4]]), tgt_out=targets)
        processed = SimpleNamespace(dataloader=lambda *args, **kwargs: [batch])
        model = LogitsModel()
        expected = model.logits.detach().clone().requires_grad_()
        total = F.cross_entropy(expected.reshape(-1, 5), targets.reshape(-1), reduction="sum")
        (total / 4).backward()
        config = replace(LuongConfig.smoke(), batch_size=4, epochs=1, optimizer="sgd",
                         learning_rate=0.1, grad_clip=100.0)
        metrics = fit(model, processed, torch.device("cpu"), config, verbose=False)
        torch.testing.assert_close(model.logits, expected.detach() - 0.1 * expected.grad)
        self.assertAlmostEqual(metrics["loss"], total.item())

    def test_embedding_gradient_uses_dense_norm_scale(self) -> None:
        model = nn.ModuleDict({"embedding": nn.Embedding(1, 1), "dense": nn.Linear(2, 1, bias=False)})
        model["embedding"].weight.grad = torch.tensor([[100.0]])
        model["dense"].weight.grad = torch.tensor([[3.0, 4.0]])
        _clip_gradients(model, 2.5)
        torch.testing.assert_close(model["dense"].weight.grad, torch.tensor([[1.5, 2.0]]))
        torch.testing.assert_close(model["embedding"].weight.grad, torch.tensor([[50.0]]))

    def test_opennmt_clipping_includes_embeddings(self) -> None:
        model = nn.ModuleDict({"embedding": nn.Embedding(1, 1), "dense": nn.Linear(2, 1, bias=False)})
        model["embedding"].weight.grad = torch.tensor([[12.0]])
        model["dense"].weight.grad = torch.tensor([[3.0, 4.0]])
        _clip_gradients(model, 6.5, convention="opennmt")
        torch.testing.assert_close(model["embedding"].weight.grad, torch.tensor([[6.0]]))
        torch.testing.assert_close(model["dense"].weight.grad, torch.tensor([[1.5, 2.0]]))

    def test_opennmt_short_batch_uses_actual_sentence_count(self) -> None:
        class LogitsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(1, 2, 5, dtype=torch.double))

            def forward(self, src, lengths, tgt):
                return self.logits

        target = torch.tensor([[4, 2]])
        batch = dict(src=torch.tensor([[4]]), src_lengths=torch.tensor([1]),
                     tgt_in=torch.tensor([[1, 4]]), tgt_out=target)
        processed = SimpleNamespace(dataloader=lambda *args, **kwargs: [batch])
        model = LogitsModel()
        expected = model.logits.detach().clone().requires_grad_()
        loss = F.cross_entropy(expected.reshape(-1, 5), target.reshape(-1), reduction="sum")
        loss.backward()
        cfg = replace(LuongConfig.smoke(), batch_size=4, optimizer="sgd", learning_rate=.1,
                      grad_clip=100., training_convention="opennmt")
        metrics = fit(model, processed, torch.device("cpu"), cfg, verbose=False)
        torch.testing.assert_close(model.logits, expected.detach() - .1 * expected.grad)
        self.assertAlmostEqual(metrics["loss"], loss.item())


if __name__ == "__main__":
    unittest.main()
