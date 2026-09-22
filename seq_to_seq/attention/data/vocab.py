from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
STANFORD_SPECIALS = frozenset({UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN})
_SPECIALS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


class Vocab:
    """Token mapping with PAD/BOS/EOS/UNK in ids 0–3."""

    def __init__(self, id_to_token: Sequence[str]) -> None:
        tokens = list(id_to_token)
        if len(tokens) < len(_SPECIALS):
            raise ValueError("vocab must include PAD, BOS, EOS, and UNK")
        if tuple(tokens[: len(_SPECIALS)]) != _SPECIALS:
            raise ValueError("vocab must start with <pad>, <s>, </s>, <unk>")
        self.id_to_token = tokens
        self.token_to_id: dict[str, int] = {}
        for index, token in enumerate(tokens):
            if token not in self.token_to_id:
                self.token_to_id[token] = index

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        unk = self.token_to_id[UNK_TOKEN]
        return [self.token_to_id.get(token, unk) for token in tokens]

    def decode(self, ids: Sequence[int], *, stop_at_eos: bool = True) -> list[str]:
        tokens: list[str] = []
        n = len(self.id_to_token)
        for token_id in ids:
            index = int(token_id)
            if index == PAD_ID:
                break
            if stop_at_eos and index == EOS_ID:
                break
            if index == BOS_ID:
                continue
            if 0 <= index < n:
                tokens.append(self.id_to_token[index])
            else:
                tokens.append(UNK_TOKEN)
        return tokens


def build_vocab(tokens: Iterable[str]) -> Vocab:
    """Specials first, then remaining types in sorted order."""
    seen = set(_SPECIALS)
    extra: set[str] = set()
    for token in tokens:
        if token not in seen:
            extra.add(token)
            seen.add(token)
    return Vocab([*_SPECIALS, *sorted(extra)])


def load_stanford_vocab(path: Path) -> Vocab:
    """Load a Stanford NMT vocab file, dropping its specials, then prepend ours."""
    if not path.is_file():
        raise FileNotFoundError(path)
    extra: list[str] = []
    seen = set(_SPECIALS)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if not token or token in STANFORD_SPECIALS or token in seen:
                continue
            extra.append(token)
            seen.add(token)
    return Vocab([*_SPECIALS, *extra])
