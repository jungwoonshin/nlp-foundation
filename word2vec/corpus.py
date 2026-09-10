from pathlib import Path


class WhitespaceCorpus:
    """Reads whitespace-tokenized text, keeping newline-separated sentences."""

    def __init__(self, path: Path, max_sentences: int | None = None) -> None:
        if max_sentences is not None and max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")
        self.path = path
        self.max_sentences = max_sentences

    def sentences(self) -> list[list[str]]:
        out: list[list[str]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                tokens = line.split()
                if not tokens:
                    continue
                out.append(tokens)
                if self.max_sentences is not None and len(out) >= self.max_sentences:
                    break
        return out

    def tokens(self) -> list[str]:
        return [token for sentence in self.sentences() for token in sentence]
