"""Immutable authors' BPE release. Cache entries are checked against recorded SHA256."""
import hashlib
import json
from pathlib import Path
import urllib.request
import uuid
from transformer.data.corpus import load_pairs

REFERENCE_COMMIT = "dc368d0af8d60270b8f4aaa2f3ac58771c551da3"
BASE_URL = f"https://raw.githubusercontent.com/LividWo/Revisit-MMT/{REFERENCE_COMMIT}/data/multi30k-en-de"
FILES = ("train.en", "train.de", "valid.en", "valid.de", "test.2016.en", "test.2016.de", "code")
EXPECTED_SPLITS = {"train": 29000, "valid": 1014, "test.2016": 1000}
EXPECTED_SHA256 = {
    "train.en": "0ffe3e3bac16e6aa33ef0dd6a7dca81587479b0d18dff0db2a92ac3f32b00c86",
    "train.de": "b2c74def7ec129f7ea7023a51501c77174c377ef06c6ccc2f031adcfb5cfadbd",
    "valid.en": "193d0d18fef5d1bc9d1cddafa13a644de3c5d6a24b45ac44476dcc5e96bfac6e",
    "valid.de": "eeb95680d6a06eafb13e3494eed9f9fa99783b892f4a3c72dd36e7ce2e000b68",
    "test.2016.en": "13b5fe3f92f78c54446d66afcaaa0a00a33ab653a8411f16812c9c5ca3795d6d",
    "test.2016.de": "375c20d50f4c486149a78dfcfb161a430a04ddabe2b7d150f0c7811dc455ac60",
    "code": "5b545f318e49f24367c7399019c9aeb5e3720b6379a08f887c2792af71c37f2a",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_multi30k(data_dir: Path, *, base_url=BASE_URL, expected_splits=None):
    expected_splits = EXPECTED_SPLITS if expected_splits is None else expected_splits
    dest = Path(data_dir) / "raw" / "multi30k.en-de"
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    entries = {}
    for name in FILES:
        path, url = dest / name, f"{base_url}/{name}"
        old = previous.get("files", {}).get(name)
        if path.exists() and path.stat().st_size > 0:
            if old and (old["url"] != url or old["sha256"] != sha256(path)):
                raise ValueError(f"cached file provenance/checksum mismatch: {name}")
        else:
            scratch = Path(__file__).resolve().parents[2] / ".tmp/transformer/downloads"
            scratch.mkdir(parents=True, exist_ok=True)
            temporary = scratch / f"{uuid.uuid4().hex}.part"
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "nlp-foundation/transformer"})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 256):
                        output.write(chunk)
                if not temporary.stat().st_size:
                    raise ValueError(f"empty download: {name}")
                if base_url == BASE_URL and sha256(temporary) != EXPECTED_SHA256[name]:
                    raise ValueError(f"download checksum mismatch: {name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        checksum = sha256(path)
        if base_url == BASE_URL and checksum != EXPECTED_SHA256[name]:
            raise ValueError(f"cached file checksum mismatch: {name}")
        entries[name] = {"url": url, "sha256": checksum, "bytes": path.stat().st_size}
    splits = {prefix: len(load_pairs(dest / f"{prefix}.en", dest / f"{prefix}.de"))
              for prefix in EXPECTED_SPLITS}
    if splits != expected_splits:
        raise ValueError(f"unexpected split counts: {splits}, expected {expected_splits}")
    manifest = {"reference_commit": REFERENCE_COMMIT, "files": entries, "split_pairs": splits,
                "preprocessing": "authors' released joint 10000-merge BPE; no retokenization",
                "reference_bleu": {"test.2016": 41.02, "scorer": "legacy Fairseq corpus BLEU-4"}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dest
