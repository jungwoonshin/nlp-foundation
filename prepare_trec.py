"""Download TREC question classification (Kim 2014) as a labeled line corpus."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

from word2vec.corpus import LABEL_PREFIX

# Harvard NLP copy of TREC, already tokenized with Kim's `clean_str(..., TREC=True)`.
TRAIN_URL = (
    "https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/"
    "data/TREC.train.all"
)
TEST_URL = (
    "https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/"
    "data/TREC.test.all"
)

# Original Li & Roth files, used if the tokenized copies are unavailable.
RAW_TRAIN_URL = (
    "https://raw.githubusercontent.com/brmson/question-classification/master/"
    "data/train_5500.label"
)
RAW_TEST_URL = (
    "https://raw.githubusercontent.com/brmson/question-classification/master/"
    "data/TREC_10.label"
)

_CLEAN = [
    (re.compile(r"[^A-Za-z0-9(),!?'`]"), " "),
    (re.compile(r"'s"), " 's"),
    (re.compile(r"'ve"), " 've"),
    (re.compile(r"n't"), " n't"),
    (re.compile(r"'re"), " 're"),
    (re.compile(r"'d"), " 'd"),
    (re.compile(r"'ll"), " 'll"),
    (re.compile(r","), " , "),
    (re.compile(r"!"), " ! "),
    (re.compile(r"\("), " ( "),
    (re.compile(r"\)"), " ) "),
    (re.compile(r"\?"), " ? "),
    (re.compile(r"\s{2,}"), " "),
]


def clean_str_trec(text: str) -> str:
    """Kim (2014) `clean_str` with `TREC=True`: keep case, split punctuation."""
    for pattern, repl in _CLEAN:
        text = pattern.sub(repl, text)
    return text.strip()


def _download(url: str, dest: Path) -> None:
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _labeled_format(path: Path) -> bool:
    if not path.is_file():
        return False
    first = path.read_text(encoding="utf-8").splitlines()
    if not first:
        return False
    tokens = first[0].split()
    return bool(tokens) and tokens[0].startswith(LABEL_PREFIX) and len(tokens[0]) > len(LABEL_PREFIX)


def _to_labeled_line(line: str) -> str | None:
    tokens = line.split()
    if not tokens:
        return None
    if tokens[0].startswith(LABEL_PREFIX) and len(tokens[0]) > len(LABEL_PREFIX):
        return line.rstrip("\n")
    label_token = tokens[0]
    if ":" in label_token:
        label = label_token.split(":", 1)[0]
        text = clean_str_trec(" ".join(tokens[1:]))
    else:
        label = label_token
        text = " ".join(tokens[1:])
    if not label or not text:
        return None
    return f"{LABEL_PREFIX}{label} {text}"


def _convert(src: Path, dst: Path) -> int:
    n_docs = 0
    with dst.open("w", encoding="utf-8", newline="\n") as out:
        for line in _read_text(src).splitlines():
            converted = _to_labeled_line(line)
            if converted is None:
                continue
            out.write(converted + "\n")
            n_docs += 1
    return n_docs


def prepare_trec(data_dir: Path) -> tuple[Path, Path]:
    """Write `trec_train.txt` and `trec_test.txt` with `__label__` class ids."""
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_out = data_dir / "trec_train.txt"
    test_out = data_dir / "trec_test.txt"

    sources = (
        (TRAIN_URL, RAW_TRAIN_URL, raw_dir / "TREC.train.all", train_out),
        (TEST_URL, RAW_TEST_URL, raw_dir / "TREC.test.all", test_out),
    )
    for tokenized_url, raw_url, raw_path, out_path in sources:
        if _labeled_format(out_path):
            continue
        if not raw_path.is_file():
            try:
                _download(tokenized_url, raw_path)
            except OSError:
                _download(raw_url, raw_path)
        n_docs = _convert(raw_path, out_path)
        print(f"wrote {n_docs:,} labeled questions -> {out_path}")
    return train_out, test_out


if __name__ == "__main__":
    prepare_trec(Path(__file__).resolve().parent / "data")
