from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from word2vec.config import ProcessingConfig, resolve_corpus_path
from word2vec.corpus import WhitespaceCorpus
from word2vec.dataset import SkipGramDataset, make_negative_collate
from word2vec.negative_sampling import NegativeSampler
from word2vec.skipgram import SkipGramPairBuilder
from word2vec.subsample import FrequentWordSubsampler
from word2vec.vocab import Vocab


@dataclass
class ProcessedCorpus:
    vocab: Vocab
    dataset: SkipGramDataset
    config: ProcessingConfig
    raw_token_count: int
    kept_token_count: int
    negative_sampler: NegativeSampler | None = None

    def dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        with_negatives: bool = True,
    ) -> DataLoader:
        collate_fn = None
        if with_negatives:
            if self.negative_sampler is None:
                raise RuntimeError(
                    "Negative sampling collate requested, but no noise table was built. "
                    "Call process(build_negative_sampler=True)."
                )
            collate_fn = make_negative_collate(self.negative_sampler, self.config.num_negatives)
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


def process_corpus(path: str | Path, config: ProcessingConfig | None = None) -> ProcessedCorpus:
    config = config or ProcessingConfig()
    config.validate()
    corpus_path = resolve_corpus_path(path)
    rng = np.random.default_rng(config.seed)

    tokens = WhitespaceCorpus(corpus_path).tokens()
    vocab = Vocab.build(tokens, min_count=config.min_count, huffman=config.build_huffman)
    encoded = vocab.encode(tokens)
    subsampled = FrequentWordSubsampler(vocab, config.subsample_threshold, rng).apply(encoded)
    centers, contexts = SkipGramPairBuilder(config.window_size, rng).build(subsampled)
    negatives = None
    if config.build_negative_sampler:
        negatives = NegativeSampler(
            vocab,
            power=config.unigram_power,
            table_size=config.negative_table_size,
            rng=rng,
        )

    return ProcessedCorpus(
        vocab=vocab,
        dataset=SkipGramDataset(centers, contexts),
        config=config,
        raw_token_count=len(tokens),
        kept_token_count=len(subsampled),
        negative_sampler=negatives,
    )
