from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from word2vec.hierarchical_softmax import HuffmanCoding


@dataclass(frozen=True)
class Vocab:
    word_to_id: dict[str, int]
    id_to_word: tuple[str, ...]
    counts: tuple[int, ...]
    coding: HuffmanCoding | None = None

    @classmethod
    def build(cls, tokens: list[str], min_count: int, *, huffman: bool = False) -> Vocab:
        frequencies = Counter(tokens)
        kept = sorted(
            ((word, count) for word, count in frequencies.items() if count >= min_count),
            key=lambda item: (-item[1], item[0]),
        )
        if not kept:
            raise ValueError(
                f"Vocabulary is empty after applying min_count={min_count}."
            )
        word_to_id = {word: index for index, (word, _) in enumerate(kept)}
        id_to_word = tuple(word for word, _ in kept)
        counts = tuple(count for _, count in kept)
        coding = HuffmanCoding.from_counts(counts) if huffman else None
        return cls(word_to_id=word_to_id, id_to_word=id_to_word, counts=counts, coding=coding)

    def __len__(self) -> int:
        return len(self.id_to_word)

    def encode(self, tokens: list[str]) -> list[int]:
        lookup = self.word_to_id
        return [lookup[token] for token in tokens if token in lookup]

    def decode(self, token_id: int) -> str:
        return self.id_to_word[token_id]

    def total_tokens(self) -> int:
        return sum(self.counts)
