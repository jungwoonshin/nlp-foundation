from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LABEL_PREFIX = "__label__"


@dataclass(frozen=True)
class Document:
    """One newline-separated document. `label` is the class id if present."""

    tokens: list[str]
    label: str | None = None


def parse_document_line(line: str) -> Document | None:
    """Split a line into optional FastText `__label__X` and remaining tokens."""
    tokens = line.split()
    if not tokens:
        return None
    if tokens[0].startswith(LABEL_PREFIX) and len(tokens[0]) > len(LABEL_PREFIX):
        return Document(tokens=tokens[1:], label=tokens[0][len(LABEL_PREFIX) :])
    return Document(tokens=tokens)


class WhitespaceCorpus:
    """Reads whitespace-tokenized text, keeping newline-separated documents."""

    def __init__(self, path: Path, max_sentences: int | None = None) -> None:
        if max_sentences is not None and max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")
        self.path = path
        self.max_sentences = max_sentences

    def documents(self) -> list[Document]:
        out: list[Document] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                document = parse_document_line(line)
                if document is None:
                    continue
                out.append(document)
                if self.max_sentences is not None and len(out) >= self.max_sentences:
                    break
        return out

    def sentences(self) -> list[list[str]]:
        return [document.tokens for document in self.documents()]

    def tokens(self) -> list[str]:
        return [token for sentence in self.sentences() for token in sentence]
