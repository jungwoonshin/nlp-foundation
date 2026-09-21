"""Download Stanford IWSLT'15 English–Vietnamese (Luong NMT small set)."""

from __future__ import annotations

from pathlib import Path

from scripts.paths import DATA_DIR
from seq_to_seq.attention.data.corpus import count_parallel_lines
from seq_to_seq.attention.data.download import download_iwslt15


def prepare_iwslt15(data_dir: Path) -> Path:
    """Fetch tokenized IWSLT'15 En–Vi files into data/raw/iwslt15.en-vi/."""
    dest = download_iwslt15(data_dir)
    splits = (
        ("train", "train.en", "train.vi"),
        ("dev tst2012", "tst2012.en", "tst2012.vi"),
        ("test tst2013", "tst2013.en", "tst2013.vi"),
    )
    for label, src_name, tgt_name in splits:
        n = count_parallel_lines(dest / src_name, dest / tgt_name)
        print(f"{label}: {n:,} sentence pairs")
    return dest


if __name__ == "__main__":
    prepare_iwslt15(DATA_DIR)
