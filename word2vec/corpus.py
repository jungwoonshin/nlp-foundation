from pathlib import Path


class WhitespaceCorpus:
    """Reads a whitespace-tokenized text file (text8-style: one stream of words)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def tokens(self) -> list[str]:
        text = self.path.read_text(encoding="utf-8")
        return text.split()
