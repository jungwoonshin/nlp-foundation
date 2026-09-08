from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from word2vec.config import ProcessingConfig
from word2vec.dataset import SkipGramDataset, make_negative_collate
from word2vec.hierarchical_softmax import HuffmanCoding, HierarchicalSoftmax
from word2vec.negative_sampling import NegativeSampler, NegativeSampling
from word2vec.pipeline import ProcessedCorpus
from word2vec.skipgram import SkipGramPairBuilder
from word2vec.subsample import FrequentWordSubsampler
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


class CbowTests(unittest.TestCase):
    def test_builder_pads_context_bags(self) -> None:
        rng = np.random.default_rng(0)
        builder = SkipGramPairBuilder(window_size=2, rng=rng)
        centers, bags = builder.build_cbow([0, 1, 2, 3, 4])
        self.assertEqual(centers.ndim, 1)
        self.assertEqual(bags.ndim, 2)
        self.assertEqual(bags.shape[0], centers.shape[0])
        self.assertEqual(bags.shape[1], 4)
        self.assertTrue((bags >= -1).all())

    def test_neg_and_hs_accept_context_bags(self) -> None:
        bag = torch.tensor([[0, 1, -1], [2, 3, 1]])
        target = torch.tensor([4, 0])
        negatives = torch.randint(0, 8, (2, 3))
        neg_model = NegativeSampling(embedding_dim=5, vocab_size=8)
        neg_loss = neg_model(bag, target, negatives)
        self.assertEqual(tuple(neg_loss.shape), ())
        neg_loss.backward()

        coding = HuffmanCoding.from_counts([5, 4, 3, 2, 1, 1, 1, 1])
        hs = HierarchicalSoftmax(coding, embedding_dim=5, vocab_size=8)
        hs_loss = hs(bag, target)
        self.assertEqual(tuple(hs_loss.shape), ())
        hs_loss.backward()

    def test_collate_excludes_center_and_bag(self) -> None:
        tokens = [chr(ord("a") + i) for i in range(20) for _ in range(3)]
        vocab = Vocab.build(tokens, min_count=1, huffman=False)
        sampler = NegativeSampler(vocab, power=0.75, table_size=200, rng=np.random.default_rng(1))
        centers = np.array([0, 1], dtype=np.int64)
        bags = np.array([[2, 3, -1], [4, 5, 6]], dtype=np.int64)
        loader = DataLoader(
            SkipGramDataset(centers, bags),
            batch_size=2,
            shuffle=False,
            collate_fn=make_negative_collate(sampler, num_negatives=6),
        )
        batch = next(iter(loader))
        for i in range(2):
            blocked = {int(batch["center"][i])}
            blocked.update(int(x) for x in batch["context"][i].tolist() if int(x) >= 0)
            sampled = set(int(x) for x in batch["negatives"][i].tolist())
            self.assertTrue(sampled.isdisjoint(blocked))


class PerEpochSubsampleTests(unittest.TestCase):
    def test_same_keep_table_new_draws_each_epoch(self) -> None:
        tokens = ["the"] * 400 + ["cat"] * 40 + ["dog"] * 40
        vocab = Vocab.build(tokens, min_count=1, huffman=False)
        ids = vocab.encode(tokens)
        corpus = ProcessedCorpus(
            vocab=vocab,
            token_ids=ids,
            subsampler=FrequentWordSubsampler(vocab, 1e-3),
            config=ProcessingConfig(seed=0, window_size=2, architecture="skipgram"),
            raw_token_count=len(tokens),
        )
        first = corpus.rebuild_examples(epoch=1)
        first_kept = corpus.kept_token_count
        second = corpus.rebuild_examples(epoch=2)
        self.assertNotEqual(first_kept, 0)
        self.assertGreater(len(first), 0)
        self.assertGreater(len(second), 0)
        same_seed = corpus.rebuild_examples(epoch=1)
        self.assertEqual(len(same_seed), len(first))
        self.assertTrue(torch.equal(same_seed.centers, first.centers))
        self.assertTrue(torch.equal(same_seed.contexts, first.contexts))
        self.assertNotEqual(
            corpus.subsampler.apply(ids, np.random.default_rng(11)),
            corpus.subsampler.apply(ids, np.random.default_rng(12)),
        )


if __name__ == "__main__":
    unittest.main()
