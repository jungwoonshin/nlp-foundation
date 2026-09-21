"""Download text8 and the Stanford Rare Words similarity set."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from scripts.paths import DATA_DIR
from scripts.word2vec.prepare import prepare_text8

RW_URL = "https://nlp.stanford.edu/~lmthang/morphoNLM/rw.zip"


def prepare_rw(data_dir: Path) -> Path:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "rw" / "rw.txt"
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path
    zip_path = raw_dir / "rw.zip"
    if not zip_path.is_file():
        print(f"downloading Rare Words -> {zip_path}")
        urllib.request.urlretrieve(RW_URL, zip_path)
    print(f"extracting {zip_path} -> {raw_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)
    if not out_path.is_file():
        raise FileNotFoundError(f"rw/rw.txt was not in {zip_path}")
    return out_path


if __name__ == "__main__":
    text8 = prepare_text8(DATA_DIR)
    rw = prepare_rw(DATA_DIR)
    print(f"text8: {text8} ({text8.stat().st_size:,} bytes)")
    print(f"rw: {rw} ({rw.stat().st_size:,} bytes)")
