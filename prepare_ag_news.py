"""Download AG News (smallest Zhang/FastText classification set) as a line corpus."""

from __future__ import annotations

import csv
import re
import urllib.request
from pathlib import Path

from word2vec.corpus import LABEL_PREFIX

# Same CSV torchtext / CharCNN Keras use (Zhang et al. 2015 split).
TRAIN_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/"
    "data/ag_news_csv/train.csv"
)
_TOKEN = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN.split(text.lower()) if tok]


def _labeled_format(out_path: Path) -> bool:
    if not out_path.is_file():
        return False
    with out_path.open(encoding="utf-8") as handle:
        first = handle.readline().split()
    return bool(first) and first[0].startswith(LABEL_PREFIX) and len(first[0]) > len(LABEL_PREFIX)


def prepare_ag_news(data_dir: Path) -> Path:
    """Fetch train.csv and write one `__label__<class> article` line per document."""
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_dir / "ag_news_train.csv"
    out_path = data_dir / "ag_news.txt"

    if not raw_csv.is_file():
        print(f"downloading AG News train -> {raw_csv}")
        urllib.request.urlretrieve(TRAIN_URL, raw_csv)

    if (
        out_path.is_file()
        and out_path.stat().st_mtime >= raw_csv.stat().st_mtime
        and _labeled_format(out_path)
    ):
        return out_path

    n_docs = 0
    with raw_csv.open(encoding="utf-8", newline="") as src, out_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for row in csv.reader(src):
            if len(row) < 3:
                continue
            label = row[0].strip().strip('"')
            if not label:
                continue
            tokens = _tokenize(f"{row[1]} {row[2]}")
            if not tokens:
                continue
            dst.write(f"{LABEL_PREFIX}{label} " + " ".join(tokens) + "\n")
            n_docs += 1
    print(f"wrote {n_docs:,} labeled articles -> {out_path}")
    return out_path


if __name__ == "__main__":
    prepare_ag_news(Path(__file__).resolve().parent / "data")
