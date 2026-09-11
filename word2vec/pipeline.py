from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from word2vec.config import ProcessingConfig, resolve_corpus_path
from word2vec.corpus import WhitespaceCorpus
from word2vec.dataset import (
    FastTextDataset,
    SkipGramDataset,
    make_negative_collate,
    feature_frequency,
    pad_fasttext_collate,
    word_ngrams,
)
from word2vec.negative_sampling import NegativeSampler
from word2vec.skipgram import SkipGramPairBuilder
from word2vec.subsample import FrequentWordSubsampler
from word2vec.vocab import Vocab


def _chunk(ids: list[int], max_length: int) -> list[list[int]]:
    return [ids[index : index + max_length] for index in range(0, len(ids), max_length)]


def _concat_windows(
    parts: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if not parts:
        raise ValueError("No windows left after sentence cuts and subsampling.")
    centers = np.concatenate([part[0] for part in parts])
    contexts = np.concatenate([part[1] for part in parts], axis=0)
    return centers, contexts


@dataclass
class ProcessedCorpus:
    vocab: Vocab
    sentences: list[list[int]]
    subsampler: FrequentWordSubsampler
    config: ProcessingConfig
    raw_token_count: int
    labels: list[str | None] | None = None
    label_to_id: dict[str, int] | None = None
    negative_sampler: NegativeSampler | None = None
    kept_token_count: int = 0
    dataset: SkipGramDataset | FastTextDataset | None = None

    def rebuild_examples(self, epoch: int) -> SkipGramDataset | FastTextDataset:
        if self.config.architecture == "fasttext":
            return self._rebuild_fasttext()
        return self._rebuild_windows(epoch)

    def _rebuild_fasttext(self) -> FastTextDataset:
        """One document: unique tokens with raw counts, label as target."""
        if self.labels is None or self.label_to_id is None:
            raise RuntimeError("FastText examples need a class label on every document.")
        documents = list(zip(self.sentences, self.labels, strict=True))
        if self.config.max_examples is not None:
            documents = documents[: self.config.max_examples]
        feature_rows: list[np.ndarray] = []
        weight_rows: list[np.ndarray] = []
        ngram_rows: list[np.ndarray] = []
        label_ids: list[int] = []
        kept_total = 0
        ngram_size = self.config.ngram_size
        for sentence, label in documents:
            if label is None:
                raise RuntimeError("FastText examples need a class label on every document.")
            kept_total += len(sentence)
            features, weights = feature_frequency(sentence)
            feature_rows.append(features)
            weight_rows.append(weights)
            ngram_rows.append(word_ngrams(sentence, ngram_size))
            label_ids.append(self.label_to_id[label])
        if not feature_rows:
            raise ValueError("No labeled documents left after vocabulary filtering.")
        self.kept_token_count = kept_total
        self.dataset = FastTextDataset(
            feature_rows,
            weight_rows,
            ngram_rows,
            label_ids,
        )
        return self.dataset

    def _rebuild_windows(self, epoch: int) -> SkipGramDataset:
        """Subsample each sentence, cut at max_sentence_length, then build windows."""
        rng = np.random.default_rng(self.config.seed + 1_000_003 * epoch)
        builder = SkipGramPairBuilder(self.config.window_size, rng)
        parts: list[tuple[np.ndarray, np.ndarray]] = []
        kept_total = 0
        max_length = self.config.max_sentence_length
        for sentence in self.sentences:
            kept = self.subsampler.apply(sentence, rng)
            kept_total += len(kept)
            for buffer in _chunk(kept, max_length):
                if len(buffer) < 2:
                    continue
                if self.config.architecture == "cbow":
                    parts.append(builder.build_cbow(buffer))
                else:
                    parts.append(builder.build(buffer))
        self.kept_token_count = kept_total
        centers, contexts = _concat_windows(parts)
        limit = self.config.max_examples
        if limit is not None:
            centers = centers[:limit]
            contexts = contexts[:limit]
        self.dataset = SkipGramDataset(centers, contexts)
        return self.dataset

    def dataloader(
        self,
        batch_size: int,
        epoch: int,
        shuffle: bool = True,
        num_workers: int = 0,
        with_negatives: bool = True,
    ) -> DataLoader:
        dataset = self.rebuild_examples(epoch)
        collate_fn = None
        if self.config.architecture == "fasttext":
            collate_fn = pad_fasttext_collate
        elif with_negatives:
            if self.negative_sampler is None:
                raise RuntimeError(
                    "Negative sampling collate requested, but no noise table was built. "
                    "Call process(build_negative_sampler=True)."
                )
            collate_fn = make_negative_collate(self.negative_sampler, self.config.num_negatives)
        return DataLoader(
            dataset,
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

    raw_documents = WhitespaceCorpus(
        corpus_path, max_sentences=config.max_sentences
    ).documents()
    vocab = Vocab.build(
        [token for document in raw_documents for token in document.tokens],
        min_count=config.min_count,
        huffman=config.build_huffman,
    )
    sentences: list[list[int]] = []
    labels: list[str | None] = []
    for document in raw_documents:
        ids = vocab.encode(document.tokens)
        if not ids:
            continue
        sentences.append(ids)
        labels.append(document.label)
    if config.architecture == "fasttext" and (not labels or any(label is None for label in labels)):
        raise ValueError("FastText classification requires a class label on every document.")
    label_to_id = None
    if labels and all(label is not None for label in labels):
        unique = sorted({label for label in labels if label is not None})
        label_to_id = {label: index for index, label in enumerate(unique)}
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
        sentences=sentences,
        subsampler=FrequentWordSubsampler(vocab, config.subsample_threshold),
        config=config,
        raw_token_count=sum(len(document.tokens) for document in raw_documents),
        labels=labels,
        label_to_id=label_to_id,
        negative_sampler=negatives,
    )
