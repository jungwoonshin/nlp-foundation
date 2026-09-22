from __future__ import annotations

import math
import unittest
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
from seq_to_seq.attention.model.local_attention import LocalAttention
from seq_to_seq.attention.model.lstm import LSTMCell, StackedLSTM
from seq_to_seq.attention.model.nmt import LuongNMT
from seq_to_seq.attention.model.scores import score_concat, score_dot, score_general, score_location
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
        kept = filter_by_length([(["a"] * 3, ["b"] * 2), (["a"] * 6, ["b"])], max_len=4)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(kept[0][0]), 3)

    def test_collate_pads_and_lengths(self) -> None:
        dataset = ParallelDataset([[4, 5, 6], [7, 8]], [[4, 5], [6]])
        batch = pad_collate([dataset[0], dataset[1]])
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


class ModelStubTests(unittest.TestCase):
    def test_luong_nmt_and_layers_raise(self) -> None:
        config = LuongConfig()
        with self.assertRaises(NotImplementedError):
            LuongNMT(12, 12, config)
        with self.assertRaises(NotImplementedError):
            LSTMCell(8, 8)
        with self.assertRaises(NotImplementedError):
            StackedLSTM(8, 8, 2)
        with self.assertRaises(NotImplementedError):
            StackedLSTMEncoder(12, 8, 8, 2)
        with self.assertRaises(NotImplementedError):
            GlobalAttention(8, "dot")
        with self.assertRaises(NotImplementedError):
            LocalAttention(8, "general", predictive=True)
        with self.assertRaises(NotImplementedError):
            AttentionalDecoder(
                12, 8, 8, 2, attention="global", score="dot", input_feeding=True
            )

    def test_score_functions_raise(self) -> None:
        h_t = torch.zeros(2, 4)
        source_h = torch.zeros(2, 3, 4)
        with self.assertRaises(NotImplementedError):
            score_dot(h_t, source_h)
        with self.assertRaises(NotImplementedError):
            score_general(h_t, source_h, torch.zeros(4, 4))
        with self.assertRaises(NotImplementedError):
            score_concat(h_t, source_h, torch.zeros(4, 8), torch.zeros(4))
        with self.assertRaises(NotImplementedError):
            score_location(h_t, 3, torch.zeros(3, 4))


if __name__ == "__main__":
    unittest.main()
