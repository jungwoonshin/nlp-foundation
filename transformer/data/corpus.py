"""Already-tokenized reference text; no source reversal or BPE retraining."""
from pathlib import Path
from itertools import zip_longest


def load_pairs(source: Path, target: Path) -> list[tuple[list[str], list[str]]]:
    pairs = []
    with source.open(encoding="utf-8") as src, target.open(encoding="utf-8") as tgt:
        for line, (s, t) in enumerate(zip_longest(src, tgt), 1):
            if s is None or t is None:
                raise ValueError(f"parallel file length mismatch at line {line}")
            if not s.split() or not t.split():
                raise ValueError(f"empty content at line {line}")
            pairs.append((s.split(), t.split()))
    if not pairs:
        raise ValueError("empty parallel corpus")
    return pairs


def smoke_pairs():
    return [("a small red house .".split(), "ein kleines rotes haus .".split()),
            ("a small blue house .".split(), "ein kleines blaues haus .".split()),
            ("a red car .".split(), "ein rotes auto .".split()),
            ("a blue car .".split(), "ein blaues auto .".split())]
