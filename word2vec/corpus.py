from pathlib import Path


class WhitespaceCorpus:
    """Reads whitespace-tokenized text, keeping newline-separated sentences."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def sentences(self) -> list[list[str]]:
        text = self.path.read_text(encoding="utf-8")
        return [line.split() for line in text.splitlines() if line.split()]

    def tokens(self) -> list[str]:
        return [token for sentence in self.sentences() for token in sentence]
