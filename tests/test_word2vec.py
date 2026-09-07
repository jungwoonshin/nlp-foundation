from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from word2vec.dataset import SkipGramDataset, make_negative_collate
from word2vec.hierarchical_softmax import HuffmanCoding, HierarchicalSoftmax
from word2vec.negative_sampling import NegativeSampler, NegativeSampling
from word2vec.vocab import Vocab


class HuffmanPathTests(unittest.TestCase):
    def test_from_counts_rejects_single_word(self) -> None:
        with self.assertRaises(ValueError):
            HuffmanCoding.from_counts([10])

    def test_frequent_words_have_shorter_or_equal_codes(self) -> None:
        counts = [100, 50, 10, 5, 1]
        coding = HuffmanCoding.from_counts(counts)
        lengths = coding.path_mask.sum(dim=1).tolist()
        self.assertEqual(coding.num_inner_nodes, len(counts) - 1)
        for rarer in range(1, len(counts)):
            self.assertLessEqual(lengths[rarer - 1], lengths[rarer])

    def test_paths_are_padded_and_prefix_free(self) -> None:
        coding = HuffmanCoding.from_counts([8, 4, 2, 1])
        vocab_size, max_len = coding.path_nodes.shape
        self.assertEqual(vocab_size, 4)
        codes = []
        for word_id in range(vocab_size):
            mask = coding.path_mask[word_id]
            length = int(mask.sum())
            self.assertGreater(length, 0)
            self.assertTrue(mask[:length].all())
            if length < max_len:
                self.assertFalse(mask[length:].any())
                self.assertTrue((coding.path_nodes[word_id, length:] == -1).all())
            bits = tuple(int(b) for b in coding.path_codes[word_id, :length].tolist())
            nodes = coding.path_nodes[word_id, :length]
            self.assertTrue(((nodes >= 0) & (nodes < coding.num_inner_nodes)).all())
            self.assertTrue(((coding.path_codes[word_id, :length] == 0) | (coding.path_codes[word_id, :length] == 1)).all())
            codes.append(bits)
        for i, left in enumerate(codes):
            for j, right in enumerate(codes):
                if i == j:
                    continue
                n = min(len(left), len(right))
                self.assertNotEqual(left[:n], right[:n])

    def test_vocab_skips_huffman_unless_requested(self) -> None:
        tokens = ["aa", "aa", "bb", "bb", "cc", "cc"]
        plain = Vocab.build(tokens, min_count=1, huffman=False)
        coded = Vocab.build(tokens, min_count=1, huffman=True)
        self.assertIsNone(plain.coding)
        self.assertIsNotNone(coded.coding)
        self.assertEqual(len(coded.coding.path_mask), len(coded))


class NegativeSamplingShapeTests(unittest.TestCase):
    def test_forward_shapes_and_backward(self) -> None:
        batch_size, num_negatives, dim, vocab_size = 4, 5, 8, 12
        model = NegativeSampling(embedding_dim=dim, vocab_size=vocab_size)
        center = torch.randint(0, vocab_size, (batch_size,))
        target = torch.randint(0, vocab_size, (batch_size,))
        negatives = torch.randint(0, vocab_size, (batch_size, num_negatives))
        loss = model(center, target, negatives)
        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.center_embeddings.weight.grad)
        self.assertIsNotNone(model.context_embedding.weight.grad)

    def test_hierarchical_softmax_batch_shapes(self) -> None:
        coding = HuffmanCoding.from_counts([20, 10, 5, 2])
        model = HierarchicalSoftmax(coding, embedding_dim=6, vocab_size=4)
        center = torch.tensor([0, 1, 2])
        target = torch.tensor([3, 0, 1])
        loss = model(center, target)
        self.assertEqual(tuple(loss.shape), ())
        loss.backward()
        self.assertIsNotNone(model.center_embeddings.weight.grad)
        self.assertIsNotNone(model.node_embeddings.weight.grad)


class CollateExcludeTests(unittest.TestCase):
    def test_negatives_are_not_center_or_context(self) -> None:
        tokens = [chr(ord("a") + i) for i in range(20) for _ in range(3)]
        vocab = Vocab.build(tokens, min_count=1, huffman=False)
        rng = np.random.default_rng(0)
        sampler = NegativeSampler(vocab, power=0.75, table_size=200, rng=rng)
        centers = np.array([0, 1, 2, 3], dtype=np.int64)
        contexts = np.array([4, 5, 6, 7], dtype=np.int64)
        dataset = SkipGramDataset(centers, contexts)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=make_negative_collate(sampler, num_negatives=8),
        )
        batch = next(iter(loader))
        negatives = batch["negatives"]
        self.assertEqual(tuple(negatives.shape), (4, 8))
        for i in range(4):
            blocked = {int(batch["center"][i]), int(batch["context"][i])}
            sampled = set(int(x) for x in negatives[i].tolist())
            self.assertTrue(sampled.isdisjoint(blocked))


if __name__ == "__main__":
    unittest.main()
