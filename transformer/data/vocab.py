"""Joined Fairseq-style dictionary: frequency order, lexical ties, padding to eight."""
from collections import Counter
from dataclasses import dataclass

BOS_ID, PAD_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]


@dataclass
class Vocab:
    tokens: list[str]
    counts: list[int]

    def __post_init__(self):
        if self.tokens[:4] != SPECIALS or len(self.tokens) != len(self.counts):
            raise ValueError("invalid vocabulary framing")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("duplicate vocabulary tokens")
        self.indices = {token: i for i, token in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    def encode(self, tokens):
        return [self.indices.get(token, UNK_ID) for token in tokens]

    def decode(self, ids, *, reference=False):
        result = []
        for value in ids:
            value = int(value)
            if value == EOS_ID:
                break
            if value in (PAD_ID, BOS_ID):
                continue
            result.append("<<unk>>" if reference and value == UNK_ID else self.tokens[value])
        return result

    def to_dict(self):
        return {"tokens": self.tokens, "counts": self.counts}


def build_vocab(pairs, padding_factor=8):
    if padding_factor < 1:
        raise ValueError("padding_factor must be positive")
    counts = Counter(token for src, tgt in pairs for row in (src, tgt) for token in row)
    for token in SPECIALS:
        counts.pop(token, None)
    ordered = sorted(counts, key=lambda token: (-counts[token], token))
    tokens, frequencies = SPECIALS + ordered, [1] * 4 + [counts[t] for t in ordered]
    i = 0
    while len(tokens) % padding_factor:
        token = f"madeupword{i:04d}"
        i += 1
        if token not in counts:
            tokens.append(token)
            frequencies.append(0)
    return Vocab(tokens, frequencies)
