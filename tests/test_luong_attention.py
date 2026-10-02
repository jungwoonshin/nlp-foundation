from __future__ import annotations

import math
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch
from torch.nn import functional as F

from seq_to_seq.attention.config import LuongConfig
from seq_to_seq.attention.data.corpus import filter_by_length, load_parallel_lines, tokenize
from seq_to_seq.attention.data.dataset import (
    ParallelDataset,
    pad_collate,
    process_pairs,
    process_parallel,
    smoke_parallel,
    trim_prediction,
)
from seq_to_seq.attention.data.download import IWSLT_FILES, download_iwslt15
from seq_to_seq.attention.data.vocab import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    build_vocab,
    load_stanford_vocab,
)
from seq_to_seq.attention.eval.bleu import corpus_bleu
from seq_to_seq.attention.eval.metrics import evaluate_bleu, evaluate_exact_match, evaluate_perplexity
from seq_to_seq.attention.model.decoder import AttentionalDecoder
from seq_to_seq.attention.model.encoder import StackedLSTMEncoder
from seq_to_seq.attention.model.global_attention import GlobalAttention
from seq_to_seq.attention.model.nmt import LuongNMT
from seq_to_seq.attention.training.dummy import DummyNMT
from seq_to_seq.attention.training.loop import fit, initialize_parameters, learning_rate_for_epoch


class LuongConfigTests(unittest.TestCase):
    def test_rejects_unknown_attention(self) -> None:
        with self.assertRaises(ValueError):
            LuongConfig(attention="bahdanau").validate()  # type: ignore[arg-type]


class VocabTests(unittest.TestCase):
    def test_specials_and_unk(self) -> None:
        vocab = build_vocab(["hello", "world", "hello"])
        self.assertEqual(vocab.id_to_token[PAD_ID], "<pad>")
        self.assertEqual(vocab.id_to_token[BOS_ID], "<s>")
        self.assertEqual(vocab.id_to_token[EOS_ID], "</s>")
        self.assertEqual(vocab.id_to_token[UNK_ID], "<unk>")
        self.assertEqual(vocab.encode(["hello", "missing"]), [vocab.token_to_id["hello"], UNK_ID])

    def test_load_stanford_drops_their_specials(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.en"
            path.write_text("<unk>\n<s>\n</s>\nhello\nworld\n", encoding="utf-8")
            vocab = load_stanford_vocab(path)
        self.assertEqual(vocab.id_to_token[:6], ["<pad>", "<s>", "</s>", "<unk>", "hello", "world"])


class ParallelDatasetTests(unittest.TestCase):
    def test_target_has_bos_eos_and_source_can_reverse(self) -> None:
        config = LuongConfig.smoke()
        pairs = [(["hello", "world"], ["xin", "chao"])]
        src_vocab = build_vocab(["hello", "world"])
        tgt_vocab = build_vocab(["xin", "chao"])
        processed = process_pairs(pairs, src_vocab, tgt_vocab, config, filter_length=False)
        item = processed.dataset[0]
        hello, world = src_vocab.encode(["hello", "world"])
        xin, chao = tgt_vocab.encode(["xin", "chao"])
        self.assertEqual(item["src"].tolist(), [world, hello])
        self.assertEqual(item["tgt_in"].tolist(), [BOS_ID, xin, chao])
        self.assertEqual(item["tgt_out"].tolist(), [xin, chao, EOS_ID])
        self.assertEqual(processed.src_text[0], ["hello", "world"])

    def test_length_filter_drops_long_pairs(self) -> None:
        kept = filter_by_length(
            [
                (["a"] * 3, ["b"] * 2),
                (["a"] * 4, ["b"] * 4),
                (["a"] * 5, ["b"]),
                (["a"], ["b"] * 5),
            ],
            max_len=4,
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(kept[0][0]), 3)
        self.assertEqual(len(kept[1][0]), 4)

    def test_collate_pads_and_lengths(self) -> None:
        dataset = ParallelDataset(
            [[4, 5, 6], [7, 8], [9, 10, 11, 12, 13]],
            [[4, 5], [6], [7]],
        )
        batch = pad_collate([dataset[0], dataset[1]])
        # Padding follows this batch's maximum (3), not the dataset maximum (5).
        self.assertEqual(tuple(batch["src"].shape), (2, 3))
        self.assertEqual(batch["src_lengths"].tolist(), [3, 2])
        self.assertEqual(int(batch["src"][1, 2]), PAD_ID)
        self.assertEqual(int(batch["tgt_in"][0, 0]), BOS_ID)

    def test_process_parallel_from_files(self) -> None:
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "train.en"
            tgt = Path(tmp) / "train.vi"
            src.write_text("hello world\n\ngood morning\n", encoding="utf-8")
            tgt.write_text("xin chao\n\nchao sang\n", encoding="utf-8")
            pairs = load_parallel_lines(src, tgt)
            self.assertEqual(len(pairs), 2)
            src_vocab = build_vocab(token for s, _ in pairs for token in s)
            tgt_vocab = build_vocab(token for _, t in pairs for token in t)
            processed = process_parallel(
                src, tgt, src_vocab, tgt_vocab, LuongConfig.smoke(), filter_length=False
            )
        self.assertEqual(len(processed.dataset), 2)
        self.assertEqual(tokenize("hello world"), ["hello", "world"])


class DownloadTests(unittest.TestCase):
    def test_skips_existing_files(self) -> None:
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            raw = data_dir / "raw" / "iwslt15.en-vi"
            raw.mkdir(parents=True)
            for name in IWSLT_FILES:
                (raw / name).write_text("x\n", encoding="utf-8")

            def boom(*args: object, **kwargs: object) -> None:
                raise AssertionError("should not download")

            with mock.patch("seq_to_seq.attention.data.download._download_file", boom):
                dest = download_iwslt15(data_dir)
            self.assertEqual(dest, raw)


class BleuTests(unittest.TestCase):
    def test_identical_hypotheses_score_100(self) -> None:
        hyp = ["the", "cat", "sat", "on", "the", "mat"]
        self.assertAlmostEqual(corpus_bleu([hyp], [hyp]), 100.0)

    def test_empty_hypothesis_scores_0(self) -> None:
        self.assertEqual(corpus_bleu([[]], [["a", "b", "c", "d"]]), 0.0)


class DummyTrainingTests(unittest.TestCase):
    def test_forward_shapes_backward_and_fit(self) -> None:
        config = LuongConfig.smoke()
        processed = smoke_parallel(config)
        model = DummyNMT(
            processed.src_vocab_size,
            processed.tgt_vocab_size,
            embed_dim=config.embed_dim,
            hidden_size=config.hidden_size,
        )
        initialize_parameters(model, config.init_range)
        batch = next(iter(processed.dataloader(2, shuffle=False)))
        logits = model(batch["src"], batch["src_lengths"], batch["tgt_in"])
        self.assertEqual(tuple(logits.shape), (2, batch["tgt_in"].shape[1], processed.tgt_vocab_size))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch["tgt_out"].reshape(-1),
            ignore_index=PAD_ID,
        )
        loss.backward()
        self.assertIsNotNone(model.out.weight.grad)

        chosen = torch.device("cpu")
        metrics = fit(
            model,
            processed,
            chosen,
            config,
            eval_processed=processed,
            eval_bleu=True,
            verbose=False,
        )
        self.assertTrue(math.isfinite(metrics["loss"]))
        self.assertGreaterEqual(metrics["bleu"], 0.0)
        self.assertGreaterEqual(evaluate_exact_match(model, processed, chosen, batch_size=2), 0.0)
        self.assertGreater(evaluate_perplexity(model, processed, chosen, batch_size=2), 1.0)
        self.assertGreaterEqual(evaluate_bleu(model, processed, chosen, batch_size=2), 0.0)
        pred = model.generate(batch["src"], batch["src_lengths"], max_len=4)
        self.assertEqual(pred.shape[0], 2)
        self.assertEqual(pred.dim(), 2)

    def test_sgd_learning_rate_halves_after_decay_start(self) -> None:
        config = LuongConfig(optimizer="sgd", learning_rate=1.0, lr_decay_start=5)
        config.validate()
        self.assertEqual(learning_rate_for_epoch(5, config), 1.0)
        self.assertEqual(learning_rate_for_epoch(6, config), 0.5)
        adam = LuongConfig.smoke()
        self.assertEqual(learning_rate_for_epoch(50, adam), adam.learning_rate)


class AttentionModelTests(unittest.TestCase):
    def test_location_attention_masks_padding_and_handles_short_batch_width(self) -> None:
        attention = GlobalAttention(2, "location", max_source_length=5)
        with torch.no_grad():
            attention.weight_a.weight.zero_()
        decoder_hidden = torch.zeros(2, 2)
        encoder_outputs = torch.tensor(
            [
                [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]],
                [[2.0, 0.0], [4.0, 0.0], [100.0, 0.0]],
            ]
        )
        result = attention(decoder_hidden, encoder_outputs, torch.tensor([3, 2]))

        self.assertIsNone(attention.weight_a.bias)
        self.assertEqual(tuple(attention.weight_a.weight.shape), (5, 2))
        self.assertIsNone(result.p_t)
        self.assertEqual(tuple(result.weights.shape), (2, 3))
        self.assertTrue(torch.allclose(result.weights.sum(dim=-1), torch.ones(2)))
        self.assertEqual(result.weights[1, 2].detach().item(), 0.0)
        self.assertTrue(
            torch.allclose(
                result.context,
                torch.tensor([[3.0, 0.0], [3.0, 0.0]]),
            )
        )

    def test_location_attention_rejects_source_beyond_fixed_capacity(self) -> None:
        attention = GlobalAttention(2, "location", max_source_length=5)
        with self.assertRaisesRegex(
            ValueError,
            "padded source length 6 exceeds location-attention capacity 5",
        ):
            attention(
                torch.zeros(1, 2),
                torch.zeros(1, 6, 2),
                torch.tensor([6]),
            )

    def test_content_attention_scores_match_equations_and_allow_longer_sources(self) -> None:
        decoder_hidden = torch.tensor([[1.0, 2.0]])
        encoder_outputs = torch.tensor(
            [[[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]
        )

        dot_attention = GlobalAttention(2, "dot", max_source_length=1)
        expected_dot = torch.einsum(
            "bi,bsi->bs", decoder_hidden, encoder_outputs
        )
        self.assertTrue(
            torch.allclose(
                dot_attention.dot_score(decoder_hidden, encoder_outputs),
                expected_dot,
            )
        )

        general_attention = GlobalAttention(2, "general", max_source_length=1)
        general_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        with torch.no_grad():
            general_attention.weight_a.weight.copy_(general_weight)
        expected_general = torch.einsum(
            "bi,ij,bsj->bs",
            decoder_hidden,
            general_weight,
            encoder_outputs,
        )
        self.assertTrue(
            torch.allclose(
                general_attention.general_score(decoder_hidden, encoder_outputs),
                expected_general,
            )
        )

        concat_attention = GlobalAttention(2, "concat", max_source_length=1)
        concat_weight = torch.tensor(
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]
        )
        concat_vector = torch.tensor([2.0, -1.0])
        with torch.no_grad():
            concat_attention.weight_a.weight.copy_(concat_weight)
            concat_attention.v_a.weight.copy_(concat_vector.unsqueeze(0))
        decoder_by_source = decoder_hidden.unsqueeze(1).expand(-1, 3, -1)
        combined = torch.cat([decoder_by_source, encoder_outputs], dim=-1)
        expected_concat = torch.einsum(
            "h,bsh->bs",
            concat_vector,
            torch.tanh(torch.einsum("hi,bsi->bsh", concat_weight, combined)),
        )
        self.assertTrue(
            torch.allclose(
                concat_attention.concat_score(decoder_hidden, encoder_outputs),
                expected_concat,
            )
        )

        for attention in (dot_attention, general_attention, concat_attention):
            with self.subTest(score=attention.score):
                result = attention(
                    decoder_hidden,
                    encoder_outputs,
                    torch.tensor([2]),
                )
                self.assertEqual(tuple(result.weights.shape), (1, 3))
                self.assertEqual(tuple(result.context.shape), (1, 2))
                self.assertEqual(result.weights[0, 2].detach().item(), 0.0)
                self.assertTrue(
                    torch.allclose(result.weights.sum(dim=-1), torch.ones(1))
                )

    def test_luong_nmt_forward_backward_and_generate(self) -> None:
        config = LuongConfig.smoke()
        processed = smoke_parallel(config)
        batch = next(iter(processed.dataloader(2, shuffle=False)))
        model = LuongNMT(processed.src_vocab_size, processed.tgt_vocab_size, config)
        self.assertEqual(model.attention.max_source_length, config.max_len)

        logits = model(batch["src"], batch["src_lengths"], batch["tgt_in"])
        self.assertEqual(
            tuple(logits.shape),
            (2, batch["tgt_in"].shape[1], processed.tgt_vocab_size),
        )
        self.assertTrue(torch.isfinite(logits).all())
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch["tgt_out"].reshape(-1),
            ignore_index=PAD_ID,
        )
        loss.backward()
        self.assertIsNotNone(model.attention.weight_a.weight.grad)
        self.assertIsNotNone(model.w_c.weight.grad)
        self.assertIsNone(model.w_c.bias)
        self.assertIsNone(model.out.bias)
        self.assertEqual(
            model.decoder.cells[0].ih.in_features,
            config.embed_dim + config.hidden_size,
        )

        no_feed_config = replace(config, input_feeding=False)
        no_feed_model = LuongNMT(
            processed.src_vocab_size,
            processed.tgt_vocab_size,
            no_feed_config,
        )
        self.assertEqual(
            no_feed_model.decoder.cells[0].ih.in_features,
            config.embed_dim,
        )
        no_feed_logits = no_feed_model(
            batch["src"], batch["src_lengths"], batch["tgt_in"]
        )
        self.assertEqual(no_feed_logits.shape, logits.shape)

        prediction = model.generate(batch["src"], batch["src_lengths"], max_len=4)
        self.assertEqual(prediction.shape[0], 2)
        self.assertGreaterEqual(prediction.shape[1], 1)
        self.assertLessEqual(prediction.shape[1], 4)

    def test_luong_nmt_supports_all_global_scores(self) -> None:
        base_config = LuongConfig.smoke()
        processed = smoke_parallel(base_config)
        batch = next(iter(processed.dataloader(2, shuffle=False)))

        for score in ("dot", "general", "concat", "location"):
            with self.subTest(score=score):
                config = replace(base_config, score=score)
                model = LuongNMT(
                    processed.src_vocab_size,
                    processed.tgt_vocab_size,
                    config,
                )
                logits = model(
                    batch["src"],
                    batch["src_lengths"],
                    batch["tgt_in"],
                )
                self.assertEqual(
                    tuple(logits.shape),
                    (2, batch["tgt_in"].shape[1], processed.tgt_vocab_size),
                )
                self.assertTrue(torch.isfinite(logits).all())
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch["tgt_out"].reshape(-1),
                    ignore_index=PAD_ID,
                )
                loss.backward()
                prediction = model.generate(
                    batch["src"], batch["src_lengths"], max_len=4
                )
                self.assertEqual(prediction.shape[0], 2)


class ModelStubTests(unittest.TestCase):
    def test_unimplemented_layers_raise(self) -> None:
        with self.assertRaises(NotImplementedError):
            StackedLSTMEncoder(12, 8, 8, 2)
        with self.assertRaises(NotImplementedError):
            AttentionalDecoder(
                12, 8, 8, 2, attention="global", score="dot", input_feeding=True
            )


if __name__ == "__main__":
    unittest.main()
