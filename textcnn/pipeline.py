from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from textcnn.config import TextCNNConfig
from textcnn.dataset import (
    PAD_ID,
    WORD_ID_OFFSET,
    SentenceDataset,
    make_pad_collate,
)
from word2vec.corpus import WhitespaceCorpus
from word2vec.vocab import Vocab


@dataclass
class ProcessedSentences:
    vocab: Vocab
    dataset: SentenceDataset
    label_to_id: dict[str, int]
    config: TextCNNConfig

    @property
    def vocab_size(self) -> int:
        """Embedding rows: pad at 0, then one row per vocab word."""
        return len(self.vocab) + WORD_ID_OFFSET

    def dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=make_pad_collate(self.config.filter_sizes, padding_idx=PAD_ID),
        )


def process_sentences(path: str | Path, config: TextCNNConfig | None = None) -> ProcessedSentences:
    config = config or TextCNNConfig()
    config.validate()
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    raw_documents = WhitespaceCorpus(
        corpus_path, max_sentences=config.max_sentences
    ).documents()
    if not raw_documents or any(document.label is None for document in raw_documents):
        raise ValueError("TextCNN classification requires a class label on every document.")

    vocab = Vocab.build(
        [token for document in raw_documents for token in document.tokens],
        min_count=config.min_count,
        huffman=False,
    )
    unique_labels = sorted(
        {document.label for document in raw_documents if document.label is not None}
    )
    label_to_id = {label: index for index, label in enumerate(unique_labels)}

    sequences: list[list[int]] = []
    labels: list[int] = []
    for document in raw_documents:
        assert document.label is not None
        ids = [token_id + WORD_ID_OFFSET for token_id in vocab.encode(document.tokens)]
        ids = ids[: config.max_length]
        if not ids:
            continue
        sequences.append(ids)
        labels.append(label_to_id[document.label])
        if config.max_examples is not None and len(sequences) >= config.max_examples:
            break
    if not sequences:
        raise ValueError("No labeled documents left after vocabulary filtering.")

    return ProcessedSentences(
        vocab=vocab,
        dataset=SentenceDataset(sequences, labels),
        label_to_id=label_to_id,
        config=config,
    )
