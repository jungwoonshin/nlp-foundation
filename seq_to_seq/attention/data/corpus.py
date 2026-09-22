from __future__ import annotations

from pathlib import Path

SMOKE_PAIRS: tuple[tuple[str, str], ...] = (
    ("hello world there friend", "xin chao ban than"),
    ("hello world good friend", "xin chao ban tot"),
    ("good morning there friend", "chao buoi sang than"),
    ("good morning good friend", "chao buoi sang tot"),
)


def tokenize(line: str) -> list[str]:
    return line.strip().split()


def load_parallel_lines(src_path: Path, tgt_path: Path) -> list[tuple[list[str], list[str]]]:
    """Load whitespace-tokenized parallel sentences; skip empty sides."""
    if not src_path.is_file():
        raise FileNotFoundError(src_path)
    if not tgt_path.is_file():
        raise FileNotFoundError(tgt_path)
    pairs: list[tuple[list[str], list[str]]] = []
    with src_path.open(encoding="utf-8") as src_handle, tgt_path.open(encoding="utf-8") as tgt_handle:
        for src_line, tgt_line in zip(src_handle, tgt_handle, strict=False):
            src_tokens = tokenize(src_line)
            tgt_tokens = tokenize(tgt_line)
            if not src_tokens or not tgt_tokens:
                continue
            pairs.append((src_tokens, tgt_tokens))
    return pairs


def count_parallel_lines(src_path: Path, tgt_path: Path) -> int:
    return len(load_parallel_lines(src_path, tgt_path))


def filter_by_length(
    pairs: list[tuple[list[str], list[str]]],
    max_len: int,
) -> list[tuple[list[str], list[str]]]:
    if max_len < 1:
        raise ValueError("max_len must be >= 1")
    return [
        (src, tgt)
        for src, tgt in pairs
        if 0 < len(src) <= max_len and 0 < len(tgt) <= max_len
    ]


def maybe_reverse(tokens: list[str], reverse: bool) -> list[str]:
    if not reverse:
        return list(tokens)
    return list(reversed(tokens))


def smoke_token_pairs() -> list[tuple[list[str], list[str]]]:
    return [(tokenize(src), tokenize(tgt)) for src, tgt in SMOKE_PAIRS]
