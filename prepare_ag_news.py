"""Download AG News (smallest Zhang/FastText classification set) as a line corpus."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from word2vec.corpus import LABEL_PREFIX

# Same CSV torchtext / CharCNN Keras use (Zhang et al. 2015 split).
_BASE = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv"
TRAIN_URL = f"{_BASE}/train.csv"
TEST_URL = f"{_BASE}/test.csv"


def normalize_text(line: str) -> str:
    """Match facebookresearch/fastText `classification-results.sh` `normalize_text`."""
    text = line.lower()
    text = f"{LABEL_PREFIX}{text}"
    text = text.replace("'", " ' ")
    text = text.replace('"', "")
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    text = text.replace("(", " ( ")
    text = text.replace(")", " ) ")
    text = text.replace("!", " ! ")
    text = text.replace("?", " ? ")
    text = text.replace(";", " ")
    text = text.replace(":", " ")
    return " ".join(text.split())


def _labeled_format(out_path: Path) -> bool:
    if not out_path.is_file():
        return False
    with out_path.open(encoding="utf-8") as handle:
        first = handle.readline().split()
    return bool(first) and first[0].startswith(LABEL_PREFIX) and len(first[0]) > len(LABEL_PREFIX)


def _convert_csv(raw_csv: Path, out_path: Path) -> int:
    n_docs = 0
    with raw_csv.open(encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            normalized = normalize_text(line.rstrip("\n\r"))
            if not normalized:
                continue
            tokens = normalized.split()
            if len(tokens) < 2 or not tokens[0].startswith(LABEL_PREFIX):
                continue
            dst.write(normalized + "\n")
            n_docs += 1
    return n_docs


def _prepare_split(data_dir: Path, url: str, raw_name: str, out_name: str) -> Path:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_dir / raw_name
    out_path = data_dir / out_name

    if not raw_csv.is_file():
        print(f"downloading AG News -> {raw_csv}")
        urllib.request.urlretrieve(url, raw_csv)

    if (
        out_path.is_file()
        and out_path.stat().st_mtime >= raw_csv.stat().st_mtime
        and _labeled_format(out_path)
    ):
        return out_path

    n_docs = _convert_csv(raw_csv, out_path)
    print(f"wrote {n_docs:,} labeled articles -> {out_path}")
    return out_path


def prepare_ag_news(data_dir: Path) -> tuple[Path, Path]:
    """Fetch Zhang train/test CSV and write FastText-normalized labeled lines."""
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = _prepare_split(data_dir, TRAIN_URL, "ag_news_train.csv", "ag_news_train.txt")
    test_path = _prepare_split(data_dir, TEST_URL, "ag_news_test.csv", "ag_news_test.txt")
    return train_path, test_path


if __name__ == "__main__":
    prepare_ag_news(Path(__file__).resolve().parent / "data")
