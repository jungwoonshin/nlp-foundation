from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from textcnn.config import TextCNNConfig
from textcnn.dataset import PAD_ID, WORD_ID_OFFSET, pad_sentence_collate
from textcnn.pipeline import process_sentences
from word2vec.corpus import LABEL_PREFIX


class TextCNNConfigTests(unittest.TestCase):
    def test_rejects_max_length_shorter_than_widest_filter(self) -> None:
        with self.assertRaises(ValueError):
            TextCNNConfig(filter_sizes=(3, 5), max_length=4).validate()


class SentencePipelineTests(unittest.TestCase):
    def test_process_shifts_word_ids_and_keeps_labels(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.txt"
            path.write_text(
                f"{LABEL_PREFIX}3 alpha beta alpha\n{LABEL_PREFIX}2 sports win\n",
                encoding="utf-8",
            )
            processed = process_sentences(
                path,
                TextCNNConfig(min_count=1, filter_sizes=(2, 3), max_length=16),
            )
        self.assertEqual(len(processed.dataset), 2)
        self.assertEqual(processed.label_to_id, {"2": 0, "3": 1})
        self.assertEqual(processed.vocab_size, len(processed.vocab) + 1)
        first = processed.dataset[0]
        alpha, beta = (processed.vocab.word_to_id[w] + WORD_ID_OFFSET for w in ("alpha", "beta"))
        self.assertEqual(first["tokens"].tolist(), [alpha, beta, alpha])
        self.assertEqual(int(first["label"]), processed.label_to_id["3"])
        self.assertGreaterEqual(min(first["tokens"].tolist()), WORD_ID_OFFSET)
        self.assertNotIn(PAD_ID, first["tokens"].tolist())
        self.assertEqual(int(processed.dataset[1]["label"]), processed.label_to_id["2"])

    def test_collate_pads_to_max_filter_width(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.txt"
            path.write_text(
                f"{LABEL_PREFIX}3 alpha beta\n{LABEL_PREFIX}2 sports win extra\n",
                encoding="utf-8",
            )
            processed = process_sentences(
                path,
                TextCNNConfig(min_count=1, filter_sizes=(3, 4, 5), max_length=16),
            )
        batch = next(iter(processed.dataloader(batch_size=2, shuffle=False)))
        self.assertEqual(tuple(batch["tokens"].shape), (2, 5))
        self.assertEqual(tuple(batch["label"].shape), (2,))
        self.assertTrue((batch["tokens"][0, 2:] == PAD_ID).all())
        self.assertEqual(int((batch["tokens"][0] != PAD_ID).sum()), 2)
        self.assertEqual(int((batch["tokens"][1] != PAD_ID).sum()), 3)

    def test_collate_pads_to_longest_sentence(self) -> None:
        short = {"tokens": torch.tensor([1, 2], dtype=torch.long), "label": torch.tensor(0)}
        long = {
            "tokens": torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.long),
            "label": torch.tensor(1),
        }
        batch = pad_sentence_collate([short, long], filter_sizes=(3, 4))
        self.assertEqual(tuple(batch["tokens"].shape), (2, 6))
        self.assertEqual(batch["tokens"][0].tolist(), [1, 2, 0, 0, 0, 0])
        self.assertEqual(batch["label"].tolist(), [0, 1])

    def test_process_rejects_unlabeled_documents(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.txt"
            path.write_text("alpha beta\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                process_sentences(path, TextCNNConfig(min_count=1))

    def test_process_truncates_to_max_length(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.txt"
            path.write_text(
                f"{LABEL_PREFIX}1 " + " ".join(f"w{i}" for i in range(20)) + "\n",
                encoding="utf-8",
            )
            processed = process_sentences(
                path,
                TextCNNConfig(min_count=1, filter_sizes=(3,), max_length=8),
            )
        self.assertEqual(int(processed.dataset[0]["tokens"].shape[0]), 8)


class TextCNNContractTests(unittest.TestCase):
    """Fails until textcnn/model.py defines TextCNN. That is expected."""

    def test_forward_shapes_and_backward(self) -> None:
        from textcnn.model import TextCNN

        batch_size, seq_len, dim, vocab_size, num_classes = 4, 7, 8, 12, 3
        filter_sizes = (3, 4, 5)
        num_filters = 6
        model = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=dim,
            num_classes=num_classes,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
            dropout=0.0,
            padding_idx=PAD_ID,
        )
        tokens = torch.randint(1, vocab_size, (batch_size, seq_len))
        tokens[:, -1] = PAD_ID
        logits = model(tokens)
        self.assertEqual(tuple(logits.shape), (batch_size, num_classes))
        self.assertTrue(torch.isfinite(logits).all())
        loss = F.cross_entropy(logits, torch.tensor([0, 1, 2, 0]))
        loss.backward()
        conv_grads = [
            param.grad
            for name, param in model.named_parameters()
            if param.requires_grad and "embed" not in name.lower()
        ]
        self.assertTrue(conv_grads, "expected trainable non-embedding parameters")
        self.assertTrue(any(grad is not None for grad in conv_grads))


if __name__ == "__main__":
    unittest.main()
