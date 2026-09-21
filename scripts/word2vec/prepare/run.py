"""Download full text8 and the word2vec analogical-reasoning questions."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from scripts.paths import DATA_DIR

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
QUESTIONS_URL = (
    "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
)


def prepare_text8(data_dir: Path) -> Path:
    """Fetch text8.zip and extract the 17M-word Wikipedia stream."""
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "text8.zip"
    out_path = raw_dir / "text8"

    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path

    if not zip_path.is_file():
        print(f"downloading text8 -> {zip_path}")
        urllib.request.urlretrieve(TEXT8_URL, zip_path)

    print(f"extracting {zip_path} -> {raw_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract("text8", path=raw_dir)
    if not out_path.is_file():
        raise FileNotFoundError(f"text8 was not in {zip_path}")
    return out_path


def prepare_questions_words(data_dir: Path) -> Path:
    """Fetch the Semantic-Syntactic Word Relationship test set."""
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "questions-words.txt"
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path
    print(f"downloading questions-words.txt -> {out_path}")
    urllib.request.urlretrieve(QUESTIONS_URL, out_path)
    return out_path


if __name__ == "__main__":
    text8 = prepare_text8(DATA_DIR)
    questions = prepare_questions_words(DATA_DIR)
    print(f"text8: {text8} ({text8.stat().st_size:,} bytes)")
    print(f"questions: {questions} ({questions.stat().st_size:,} bytes)")
