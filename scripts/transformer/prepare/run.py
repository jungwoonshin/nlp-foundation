"""Usage: python prepare.py transformer [--data-dir PATH]."""
import argparse
from pathlib import Path
from scripts.paths import DATA_DIR
from transformer.data.download import download_multi30k


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    directory = download_multi30k(args.data_dir)
    print(directory)
    print((directory / "manifest.json").read_text(encoding="utf-8"))
    return directory


if __name__ == "__main__":
    main()
