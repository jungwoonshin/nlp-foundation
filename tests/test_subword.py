from __future__ import annotations

import unittest

import torch

from word2vec.subword.model import SubwordNegativeSampling, character_ngrams
from word2vec.vocab import Vocab


class CharacterNgramTests(unittest.TestCase):
    def test_where_trigrams_match_the_paper(self) -> None:
        grams = character_ngrams("where", minn=3, maxn=3)
        self.assertEqual(grams, ["<wh", "whe", "her", "ere", "re>"])


class SubwordEncodeTests(unittest.TestCase):
    def test_encode_averages_word_row_and_ngrams(self) -> None:
        vocab = Vocab.build(["cat"] * 3 + ["dog"] * 3, min_count=1)
        model = SubwordNegativeSampling(embedding_dim=4, vocab=vocab, num_buckets=32)
        cat = vocab.word_to_id["cat"]
        buckets = model.subwordifier.hash_word("cat")
        self.assertGreater(len(buckets), 0)
        with torch.no_grad():
            model.subwordifier.center_embeddings.weight.zero_()
            model.subwordifier.subword_embeddings.weight.zero_()
            model.subwordifier.center_embeddings.weight[cat] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            for bucket in buckets:
                model.subwordifier.subword_embeddings.weight[bucket] = torch.tensor(
                    [0.0, 1.0, 0.0, 0.0]
                )
            encoded = model.subwordifier.encode(torch.tensor([cat]))
            composed = model.compose("cat")
        n_grams = len(buckets)
        expected = torch.tensor([1.0 / (1 + n_grams), n_grams / (1 + n_grams), 0.0, 0.0])
        self.assertTrue(torch.allclose(encoded.squeeze(0), expected, atol=1e-5))
        self.assertTrue(torch.allclose(composed, expected, atol=1e-5))

    def test_oov_uses_ngrams_only(self) -> None:
        vocab = Vocab.build(["cat"] * 3 + ["dog"] * 3, min_count=1)
        model = SubwordNegativeSampling(embedding_dim=3, vocab=vocab, num_buckets=32)
        unknown = "cats"
        buckets = model.subwordifier.hash_word(unknown)
        self.assertGreater(len(buckets), 0)
        with torch.no_grad():
            model.subwordifier.subword_embeddings.weight.zero_()
            one = torch.tensor([1.0, 2.0, 3.0])
            for bucket in buckets:
                model.subwordifier.subword_embeddings.weight[bucket] = one
            composed = model.compose(unknown)
        self.assertTrue(torch.allclose(composed, one, atol=1e-5))

    def test_init_like_fasttext_bounds(self) -> None:
        vocab = Vocab.build(["aa"] * 2 + ["bb"] * 2, min_count=1)
        dim = 8
        model = SubwordNegativeSampling(embedding_dim=dim, vocab=vocab, num_buckets=16)
        bound = 1.0 / dim
        self.assertTrue(torch.all(model.subwordifier.center_embeddings.weight.abs() <= bound + 1e-6))
        self.assertTrue(torch.all(model.context_embedding.weight == 0))


if __name__ == "__main__":
    unittest.main()
