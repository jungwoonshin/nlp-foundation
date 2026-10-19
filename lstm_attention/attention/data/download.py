from __future__ import annotations

import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

STANFORD_BASE_URL = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi"
PADDLE_ARCHIVE_URL = "https://bj.bcebos.com/paddlenlp/datasets/iwslt15.en-vi.tar.gz"
IWSLT_DIRNAME = "iwslt15.en-vi"
IWSLT_FILES: tuple[str, ...] = (
    "train.en",
    "train.vi",
    "tst2012.en",
    "tst2012.vi",
    "tst2013.en",
    "tst2013.vi",
    "vocab.en",
    "vocab.vi",
)
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def iwslt_raw_dir(data_dir: Path) -> Path:
    return data_dir / "raw" / IWSLT_DIRNAME


def _all_present(dest_dir: Path) -> bool:
    return all((dest_dir / name).is_file() and (dest_dir / name).stat().st_size > 0 for name in IWSLT_FILES)


def _download_file(url: str, dest: Path) -> None:
    print(f"downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(256 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded empty file: {dest}")


def _extract_tar(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, object] = {}
    if sys.version_info >= (3, 12):
        kwargs["filter"] = "data"
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, **kwargs)


def _flatten_extracted(root: Path, dest_dir: Path) -> None:
    """Copy IWSLT filenames into dest_dir if the archive nested them."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name in IWSLT_FILES:
            found[path.name] = path
    missing = [name for name in IWSLT_FILES if name not in found]
    if missing:
        raise RuntimeError(f"archive is missing files: {missing}")
    for name, path in found.items():
        target = dest_dir / name
        if path.resolve() == target.resolve():
            continue
        target.write_bytes(path.read_bytes())


def _download_from_stanford(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in IWSLT_FILES:
        dest = dest_dir / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        _download_file(f"{STANFORD_BASE_URL}/{name}", dest)


def _download_from_paddle(data_dir: Path, dest_dir: Path) -> None:
    archive = data_dir / "raw" / "iwslt15.en-vi.tar.gz"
    extract_root = data_dir / "raw" / "_iwslt15_extract"
    if not archive.is_file() or archive.stat().st_size == 0:
        _download_file(PADDLE_ARCHIVE_URL, archive)
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    _extract_tar(archive, extract_root)
    _flatten_extracted(extract_root, dest_dir)
    shutil.rmtree(extract_root, ignore_errors=True)


def download_iwslt15(data_dir: Path) -> Path:
    """Fetch IWSLT'15 En–Vi into data/raw/iwslt15.en-vi/.

    Tries Stanford first, then the PaddleNLP tarball (same preprocessed files)
    because nlp.stanford.edu often returns 403.
    """
    dest_dir = iwslt_raw_dir(data_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if _all_present(dest_dir):
        return dest_dir
    try:
        _download_from_stanford(dest_dir)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        print(f"Stanford download failed ({exc}); using PaddleNLP mirror.")
    if _all_present(dest_dir):
        return dest_dir
    _download_from_paddle(data_dir, dest_dir)
    if not _all_present(dest_dir):
        raise RuntimeError(f"IWSLT'15 files missing after download: {dest_dir}")
    return dest_dir
