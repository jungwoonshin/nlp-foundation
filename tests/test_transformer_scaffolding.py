"""Infrastructure checks; the real Transformer remains an intentional stub."""
import copy
from dataclasses import replace
import importlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import torch
from torch import nn

from scripts.dispatch import available_models
from scripts.paths import REPO_ROOT
from transformer.config import TransformerConfig
from transformer.data.corpus import load_pairs, smoke_pairs
from transformer.data.dataset import process_pairs, pad_collate, TokenBatchSampler
from transformer.data.download import FILES, download_multi30k
from transformer.data.masks import causal_mask, padding_mask, decoder_mask
from transformer.data.vocab import build_vocab, PAD_ID, EOS_ID, UNK_ID
from transformer.eval.bleu import corpus_bleu, remove_bpe
from transformer.eval.decoding import generate
from transformer.eval.metrics import evaluate_perplexity
from transformer.model.attention import scaled_dot_product_attention
from transformer.training.checkpoints import save_checkpoint, load_checkpoint, average_checkpoints
from transformer.training.loop import fit, train_update
from transformer.training.loss import token_loss
from transformer.training.schedule import learning_rate
from transformer.training.smoke import DummyTransformer


class TransformerScaffoldingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.scratch = REPO_ROOT / ".tmp/transformer-scaffold-tests"
        cls.scratch.mkdir(parents=True, exist_ok=True)

    def temporary_directory(self):
        context = tempfile.TemporaryDirectory(dir=self.scratch)
        self.addCleanup(context.cleanup)
        return Path(context.name)

    def processed(self):
        pairs = smoke_pairs()
        return process_pairs(pairs, build_vocab(pairs))

    def test_configuration_validation_and_smoke_defaults(self):
        TransformerConfig().validate()
        TransformerConfig.smoke().validate()
        for overrides in ({"heads": 3}, {"max_tokens": 0}, {"seed": -1},
                          {"label_smoothing": 1.}, {"adam_eps": float("nan")},
                          {"warmup_init_lr": .1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(TransformerConfig(), **overrides).validate()

    def test_joined_vocab_lexical_ties_and_no_eval_leakage(self):
        vocab = build_vocab([(["z", "a", "a"], ["b", "z"])])
        self.assertEqual(vocab.tokens[:8], ["<s>", "<pad>", "</s>", "<unk>", "a", "z", "b", "madeupword0000"])
        self.assertEqual(vocab.encode(["validation_only"]), [UNK_ID])
        self.assertEqual(len(vocab) % 8, 0)

    def test_source_eos_left_padding_target_shift_and_masking(self):
        pairs = [(["a"], ["x", "y"]), (["a", "b", "c"], ["x"])]
        processed = process_pairs(pairs, build_vocab(pairs))
        batch = pad_collate([processed.dataset[i] for i in (0, 1)])
        self.assertEqual(batch["src_lengths"].tolist(), [2, 4])
        self.assertEqual(batch["src"][0, :2].tolist(), [PAD_ID, PAD_ID])
        self.assertEqual(batch["src"][:, -1].tolist(), [EOS_ID, EOS_ID])
        self.assertEqual(batch["tgt_in"][:, 0].tolist(), [EOS_ID, EOS_ID])
        self.assertEqual(batch["tgt_out"][1, 1:].tolist(), [EOS_ID, PAD_ID])
        mask = decoder_mask(batch["tgt_in"])
        self.assertTrue(mask[0, 0, 0, 1])
        self.assertFalse(mask[0, 0, 1, 1])
        self.assertTrue(mask[1, 0, 2, 2])
        self.assertEqual(padding_mask(batch["src"])[0, 0, 0].tolist(), [True, True, False, False])

    def test_empty_and_overlong_rows_rejected_without_truncation(self):
        vocab = build_vocab(smoke_pairs())
        for pairs in ([([], ["x"])], [(["a"], [])], []):
            with self.assertRaises(ValueError):
                process_pairs(pairs, vocab)
        with self.assertRaises(ValueError):
            process_pairs([(["a"] * 5, ["b"])], vocab, max_positions=5)
        self.assertTrue(decoder_mask(torch.full((1, 3), PAD_ID)).all())
        self.assertEqual(causal_mask(3)[0, 0].sum().item(), 3)

    def test_parallel_alignment_and_empty_content(self):
        folder = self.temporary_directory()
        src, tgt = folder / "src", folder / "tgt"
        src.write_text("one\ntwo\n", encoding="utf-8")
        tgt.write_text("eins\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            load_pairs(src, tgt)
        tgt.write_text("eins\n\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            load_pairs(src, tgt)

    def test_token_budget_covers_all_rows_and_preserves_eval_order(self):
        p = self.processed()
        sampler = TokenBatchSampler(p.dataset, 15, seed=11)
        batches = list(sampler)
        self.assertEqual(sorted(i for b in batches for i in b), list(range(4)))
        self.assertEqual(batches, list(sampler))
        for indices in batches:
            batch = pad_collate([p.dataset[i] for i in indices])
            self.assertLessEqual(max(batch["src"].size(1), batch["tgt_in"].size(1)) * len(indices), 15)
        self.assertEqual([i for b in TokenBatchSampler(p.dataset, 15, shuffle=False) for i in b], list(range(4)))
        with self.assertRaises(ValueError):
            list(TokenBatchSampler(p.dataset, 1))

    def test_cached_download_checksums_and_counts(self):
        folder = self.temporary_directory()
        fixture = folder / "fixture"
        fixture.mkdir()
        for name in FILES:
            (fixture / name).write_text("#version: 0.2\na b\n" if name == "code" else "one two\n", encoding="utf-8")
        expected = {"train": 1, "valid": 1, "test.2016": 1}
        directory = download_multi30k(folder / "output", base_url=fixture.as_uri(), expected_splits=expected)
        with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            download_multi30k(folder / "output", base_url=fixture.as_uri(), expected_splits=expected)
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["split_pairs"], expected)
        (directory / "train.en").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            download_multi30k(folder / "output", base_url=fixture.as_uri(), expected_splits=expected)

    def test_failed_download_cleans_partial_file(self):
        folder = self.temporary_directory()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaises(urllib.error.URLError):
                download_multi30k(folder)
        self.assertEqual(list(folder.rglob("*.part")), [])

    def test_smoothed_loss_matches_hand_calculation_excluding_pad(self):
        probabilities = torch.tensor([[[.1, .2, .3, .4], [.4, .3, .2, .1]]], dtype=torch.double)
        logits = probabilities.log().requires_grad_()
        loss, nll, count = token_loss(logits, torch.tensor([[3, PAD_ID]]), .1)
        expected = .9 * -math.log(.4) + .1 / 4 * -probabilities[0, 0].log().sum().item()
        self.assertAlmostEqual(loss.item(), expected)
        self.assertAlmostEqual(nll.item(), -math.log(.4))
        self.assertEqual(count, 1)
        loss.backward()
        self.assertEqual(logits.grad[0, 1].abs().sum().item(), 0.)

    def test_perplexity_uses_unsmoothed_nll(self):
        p = self.processed()
        class Uniform(nn.Module):
            def forward(self, src, lengths, tgt):
                return torch.zeros(*tgt.shape, len(p.vocab))
        result = evaluate_perplexity(Uniform(), p, "cpu", epsilon=.1)
        self.assertAlmostEqual(result["perplexity"], len(p.vocab), places=4)
        self.assertEqual(result["tokens"], sum(len(tgt) + 1 for _, tgt in p.dataset.rows))

    def test_schedule_boundaries(self):
        c = TransformerConfig()
        self.assertEqual(learning_rate(0, c), 1e-7)
        self.assertAlmostEqual(learning_rate(2000, c), .005)
        self.assertAlmostEqual(learning_rate(8000, c), .0025)
        self.assertEqual(learning_rate(9000, TransformerConfig.smoke()), .001)

    def test_unequal_microbatches_equal_combined_token_update(self):
        p = self.processed()
        torch.manual_seed(10)
        split_model = DummyTransformer(len(p.vocab)).double()
        combined_model = copy.deepcopy(split_model)
        batches = [pad_collate([p.dataset[0]]), pad_collate([p.dataset[i] for i in (1, 2, 3)])]
        config = replace(TransformerConfig.smoke(), label_smoothing=.1, grad_clip=1000.)
        split_opt = torch.optim.SGD(split_model.parameters(), lr=.1)
        combined_opt = torch.optim.SGD(combined_model.parameters(), lr=.1)
        train_update(split_model, batches, split_opt, config, "cpu")
        train_update(combined_model, [pad_collate([p.dataset[i] for i in range(4)])], combined_opt, config, "cpu")
        for a, b in zip(split_model.parameters(), combined_model.parameters()):
            torch.testing.assert_close(a, b)

    def test_checkpoint_roundtrip_and_random_state(self):
        folder = self.temporary_directory()
        model = DummyTransformer(8)
        opt = torch.optim.Adam(model.parameters())
        torch.manual_seed(11)
        save_checkpoint(folder / "one.pt", model, opt, {"updates": 3}, {"config": {}})
        expected = torch.rand(5)
        with torch.no_grad():
            model.output.weight.zero_()
        loaded = load_checkpoint(folder / "one.pt", model=model, optimizer=opt, restore_rng=True)
        self.assertEqual(loaded["state"]["updates"], 3)
        torch.testing.assert_close(torch.rand(5), expected)
        self.assertGreater(model.output.weight.abs().sum().item(), 0.)

    def test_resume_with_dropout_matches_uninterrupted_training(self):
        p, folder = self.processed(), self.temporary_directory()
        class DropoutDummy(DummyTransformer):
            def forward(self, src, lengths, tgt):
                return self.output(torch.nn.functional.dropout(self.embedding(tgt), .2, self.training))
        torch.manual_seed(123)
        initial = DropoutDummy(len(p.vocab))
        whole, resumed = copy.deepcopy(initial), copy.deepcopy(initial)
        config = replace(TransformerConfig.smoke(), max_epochs=2)
        torch.manual_seed(45)
        full_state = fit(whole, p, p, config, "cpu", output_dir=folder / "whole", verbose=False)
        torch.manual_seed(45)
        fit(resumed, p, p, replace(config, max_epochs=1), "cpu", output_dir=folder / "split", verbose=False)
        state = fit(resumed, p, p, config, "cpu", output_dir=folder / "split",
                    resume=folder / "split/checkpoint_epoch_0001.pt", verbose=False)
        self.assertEqual(state["updates"], full_state["updates"])
        for a, b in zip(whole.parameters(), resumed.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        saved = load_checkpoint(folder / "split/checkpoint_epoch_0002.pt")
        self.assertEqual(saved["metadata"]["config"]["max_epochs"], 2)

    def test_mid_epoch_resume_matches_uninterrupted(self):
        p, folder = self.processed(), self.temporary_directory()
        torch.manual_seed(123)
        initial = DummyTransformer(len(p.vocab))
        whole, resumed = copy.deepcopy(initial), copy.deepcopy(initial)
        config = TransformerConfig.smoke()
        fit(whole, p, p, config, "cpu", output_dir=folder / "whole", verbose=False)
        fit(resumed, p, p, replace(config, max_updates=1), "cpu", output_dir=folder / "split", verbose=False)
        state = fit(resumed, p, p, config, "cpu", output_dir=folder / "split",
                    resume=folder / "split/checkpoint_partial.pt", verbose=False)
        self.assertEqual(state["updates"], 2)
        for a, b in zip(whole.parameters(), resumed.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_checkpoint_average_and_insufficient_history(self):
        folder = self.temporary_directory()
        model = nn.Linear(1, 1, bias=False)
        opt = torch.optim.Adam(model.parameters())
        paths = []
        for i in range(10):
            with torch.no_grad():
                model.weight.fill_(i)
            path = folder / f"{i}.pt"
            save_checkpoint(path, model, opt, {"updates": i}, {"vocab": ["same"]})
            paths.append(path)
        output = average_checkpoints(paths, folder / "average.pt")
        self.assertEqual(load_checkpoint(output)["model"]["weight"].item(), 4.5)
        with self.assertRaisesRegex(ValueError, "at least 10"):
            average_checkpoints(paths[:9], folder / "bad.pt")

    def test_average_rejects_different_vocabularies(self):
        folder = self.temporary_directory()
        model = nn.Linear(1, 1)
        opt = torch.optim.Adam(model.parameters())
        paths = [folder / "one.pt", folder / "two.pt"]
        for path, vocab in zip(paths, (["one"], ["two"])):
            save_checkpoint(path, model, opt, {}, {"vocab": vocab})
        with self.assertRaisesRegex(ValueError, "different metadata"):
            average_checkpoints(paths, folder / "average.pt", count=2)

    def test_beam_search_finds_better_complete_sequence_than_greedy(self):
        class Scripted(nn.Module):
            def forward(self, src, lengths, tgt):
                logits = torch.full((*tgt.shape, 7), -100.)
                if tgt.size(1) == 1:
                    logits[:, -1, 4] = math.log(.6)
                    logits[:, -1, 5] = math.log(.4)
                else:
                    for i in range(tgt.size(0)):
                        probability = .1 if tgt[i, -1] == 4 else .9
                        logits[i, -1, EOS_ID] = math.log(probability)
                        logits[i, -1, 6] = math.log(1 - probability)
                return logits
        src, lengths = torch.tensor([[4, EOS_ID]]), torch.tensor([2])
        self.assertEqual(generate(Scripted(), src, lengths, beam_size=1, max_tokens=1).tolist(), [[4, EOS_ID]])
        self.assertEqual(generate(Scripted(), src, lengths, beam_size=2, max_tokens=1).tolist(), [[5, EOS_ID]])

    def test_eos_minimum_and_batch_output_padding(self):
        class Stop(nn.Module):
            def forward(self, src, lengths, tgt):
                logits = torch.full((*tgt.shape, 6), -100.)
                logits[:, -1, EOS_ID] = 10.
                logits[:, -1, 4] = 9.
                return logits
        result = generate(Stop(), torch.tensor([[4, EOS_ID], [5, EOS_ID]]), torch.tensor([2, 2]), beam_size=1)
        self.assertEqual(result.tolist(), [[4, EOS_ID], [4, EOS_ID]])

    def test_bleu_case_bpe_unknown_and_brevity(self):
        sentence = "a small red house .".split()
        self.assertAlmostEqual(corpus_bleu([sentence], [sentence]), 100.)
        self.assertEqual(corpus_bleu([[]], [sentence]), 0.)
        self.assertEqual(corpus_bleu([["A", "SMALL", "RED", "HOUSE"]], [sentence]), 0.)
        self.assertEqual(corpus_bleu([["<unk>"] * 4], [["<unk>"] * 4]), 0.)
        self.assertEqual(remove_bpe(["ein", "kle@@", "ines", "haus"]), ["ein", "kleines", "haus"])
        self.assertAlmostEqual(corpus_bleu([sentence[:4]], [sentence]), 100 * math.exp(1 - 5 / 4))

    def test_all_model_constructors_are_stubs_and_scripts_discovered(self):
        modules = {
            "embeddings": [("TokenEmbedding", (8, 16))],
            "positional_encoding": [("SinusoidalPositionalEncoding", (16,))],
            "attention": [("MultiHeadAttention", (16, 2))],
            "feed_forward": [("PositionwiseFeedForward", (16, 32))],
            "normalization": [("LayerNorm", (16,)), ("ResidualConnection", (16,))],
            "encoder": [("EncoderLayer", (TransformerConfig(),)), ("Encoder", (TransformerConfig(),))],
            "decoder": [("DecoderLayer", (TransformerConfig(),)), ("Decoder", (TransformerConfig(),))],
            "nmt": [("TransformerNMT", (8, TransformerConfig()))],
        }
        for module, constructors in modules.items():
            for name, args in constructors:
                with self.subTest(name=name), self.assertRaises(NotImplementedError):
                    getattr(importlib.import_module(f"transformer.model.{module}"), name)(*args)
        with self.assertRaises(NotImplementedError):
            scaled_dot_product_attention(None, None, None)
        for stage in ("prepare", "train", "eval"):
            self.assertIn("transformer", available_models(stage))


if __name__ == "__main__":
    unittest.main()
