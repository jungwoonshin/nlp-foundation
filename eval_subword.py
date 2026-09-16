"""Train skip-gram FastText (SISG) on text8 and score Stanford Rare Words.

Compares this repo's subword model to Bojanowski et al. 2017, Figure 1 / §5.4:
SISG on ~1% of English Wikipedia, Spearman ρ×100 = 45 on RW.
text8 is Mahoney-processed Wikipedia of similar size (17M tokens).
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from functools import partial
from pathlib import Path

import torch

from prepare_text8 import prepare_text8
from word2vec.config import ProcessingConfig
from word2vec.pipeline import process_corpus
from word2vec.similarity import load_rw, spearman_from_vectors
from word2vec.subword.model import SubwordNegativeSampling
from word2vec.training import device, fit, log_corpus, neg_loss
from word2vec.vocab import Vocab

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RW_URL = "https://nlp.stanford.edu/~lmthang/morphoNLM/rw.zip"
CHECKPOINT = DATA_DIR / "raw" / "sisg_text8.pt"

# Bojanowski et al. 2017 §4.3 (SISG), evaluated as in §5.4 on RW.
EMBEDDING_DIM = 300
WINDOW_SIZE = 5
NUM_NEGATIVES = 5
MIN_COUNT = 5
SUBSAMPLE_THRESHOLD = 1e-4
UNIGRAM_POWER = 0.5
NUM_BUCKETS = 2_000_000
EPOCHS = 5
BATCH_SIZE = 8192
LEARNING_RATE = 0.05
SEED = 42
PAPER_RW_RHO_X100 = 45.0


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


def _sum_neg_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    chosen_device: torch.device,
    architecture: str,
) -> torch.Tensor:
    """Undo batch-mean so SGD lr=0.05 matches per-example FastText updates."""
    mean_loss = neg_loss(model, batch, chosen_device, architecture)
    return mean_loss * int(batch["center"].shape[0])


@torch.no_grad()
def evaluate_rw(model: SubwordNegativeSampling, pairs, title: str) -> float:
    model.eval()
    left = torch.stack([model.compose(pair.a) for pair in pairs])
    right = torch.stack([model.compose(pair.b) for pair in pairs])
    scores = torch.tensor([pair.score for pair in pairs], device=left.device)
    report = spearman_from_vectors(left, right, scores)
    print(
        f"{title}:  Spearman ρ×100 = {report.rho_x100:.1f}  "
        f"(scored {report.scored}/{report.scored + report.skipped}, "
        f"skip {report.skipped})"
    )
    print(f"  paper SISG on 1% Wikipedia RW: {PAPER_RW_RHO_X100:.0f}")
    model.train()
    return report.rho_x100


def save_checkpoint(
    model: SubwordNegativeSampling,
    vocab: Vocab,
    path: Path,
    *,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "id_to_word": vocab.id_to_word,
        "embedding_dim": model.subwordifier.embedding_dim,
        "num_buckets": model.subwordifier.num_buckets,
        "minn": model.subwordifier.minn,
        "maxn": model.subwordifier.maxn,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"saved checkpoint -> {path}")


def load_checkpoint(path: Path, chosen: torch.device) -> SubwordNegativeSampling:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    vocab = Vocab(
        word_to_id={word: index for index, word in enumerate(payload["id_to_word"])},
        id_to_word=tuple(payload["id_to_word"]),
        counts=tuple(0 for _ in payload["id_to_word"]),
    )
    model = SubwordNegativeSampling(
        embedding_dim=int(payload["embedding_dim"]),
        vocab=vocab,
        num_buckets=int(payload["num_buckets"]),
        minn=int(payload.get("minn", 3)),
        maxn=int(payload.get("maxn", 6)),
    )
    model.load_state_dict(payload["state_dict"])
    print(f"loaded checkpoint <- {path}")
    return model.to(chosen)


def train_ours(corpus_path: Path, chosen: torch.device, pairs) -> SubwordNegativeSampling:
    torch.manual_seed(SEED)
    processed = process_corpus(
        corpus_path,
        ProcessingConfig(
            min_count=MIN_COUNT,
            window_size=WINDOW_SIZE,
            subsample_threshold=SUBSAMPLE_THRESHOLD,
            num_negatives=NUM_NEGATIVES,
            unigram_power=UNIGRAM_POWER,
            seed=SEED,
            build_huffman=False,
            build_negative_sampler=True,
            architecture="skipgram",
        ),
    )
    log_corpus(processed, corpus_path)
    print(f"ngram buckets: {NUM_BUCKETS:,}  dim: {EMBEDDING_DIM}  ns power: {UNIGRAM_POWER}")
    model = SubwordNegativeSampling(
        embedding_dim=EMBEDDING_DIM,
        vocab=processed.vocab,
        num_buckets=NUM_BUCKETS,
    ).to(chosen)

    def after_epoch(epoch: int, trained: torch.nn.Module, mean_loss: float) -> None:
        # fit() multiplies a summed loss by batch size again; show per-example loss.
        print(f"epoch {epoch:3d}  display_loss={mean_loss:.4f} (summed, not per-example)")
        rho = evaluate_rw(trained, pairs, title=f"ours SISG epoch {epoch}")
        save_checkpoint(
            trained,
            processed.vocab,
            CHECKPOINT,
            extra={"epoch": epoch, "rw_rho_x100": rho},
        )

    fit(
        model,
        processed,
        chosen,
        partial(_sum_neg_loss, architecture="skipgram"),
        with_negatives=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        optimizer_name="sgd",
        min_learning_rate=1e-6,
        log_every=500,
        after_epoch=after_epoch,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and load the saved SISG checkpoint.",
    )
    args = parser.parse_args()

    corpus_path = prepare_text8(DATA_DIR)
    rw_path = prepare_rw(DATA_DIR)
    pairs = load_rw(rw_path)
    print(f"RW pairs: {len(pairs):,}")
    chosen = device()

    if args.eval_only:
        model = load_checkpoint(CHECKPOINT, chosen)
        evaluate_rw(model, pairs, title="ours SISG")
        return

    model = train_ours(corpus_path, chosen, pairs)
    evaluate_rw(model, pairs, title="ours SISG final")


if __name__ == "__main__":
    main()
