"""Train skip-gram NEG on full text8 and score questions-words.txt.

Compares this repo's implementation to a C-compatible reference (gensim, or
official word2vec if a binary is on PATH) using the same hyperparameters.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

import torch

from scripts.paths import DATA_DIR
from scripts.word2vec.prepare import prepare_questions_words, prepare_text8
from word2vec.analogy import evaluate_analogies, format_report, load_questions
from word2vec.negative_sampling.model import NegativeSampling
from word2vec.training import device, fit, log_corpus, neg_loss, process
from word2vec.vocab import Vocab

EMBEDDING_DIM = 100
WINDOW_SIZE = 5
NUM_NEGATIVES = 5
MIN_COUNT = 5
SUBSAMPLE_THRESHOLD = 1e-4
EPOCHS = 5
BATCH_SIZE = 1024
LEARNING_RATE = 0.025
SEED = 42
RESTRICT_VOCAB = 30_000


def _sum_neg_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    chosen_device: torch.device,
    architecture: str,
) -> torch.Tensor:
    """Undo batch-mean so SGD lr=0.025 matches per-example C updates."""
    mean_loss = neg_loss(model, batch, chosen_device, architecture)
    return mean_loss * int(batch["center"].shape[0])


def train_ours(corpus_path: Path, chosen: torch.device) -> tuple[NegativeSampling, Vocab]:
    torch.manual_seed(SEED)
    processed = process(
        corpus_path,
        min_count=MIN_COUNT,
        window_size=WINDOW_SIZE,
        subsample_threshold=SUBSAMPLE_THRESHOLD,
        num_negatives=NUM_NEGATIVES,
        seed=SEED,
        build_huffman=False,
        build_negative_sampler=True,
        architecture="skipgram",
    )
    log_corpus(processed, corpus_path)
    model = NegativeSampling(
        embedding_dim=EMBEDDING_DIM,
        vocab_size=len(processed.vocab),
    ).to(chosen)
    model.init_like_word2vec()
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
        min_learning_rate=1e-4,
    )
    return model, processed.vocab


CHECKPOINT = DATA_DIR / "raw" / "skipgram_neg_text8.pt"


def load_checkpoint(path: Path) -> tuple[torch.Tensor, dict[str, int]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    embeddings = payload["embeddings"]
    word_to_id = {word: index for index, word in enumerate(payload["id_to_word"])}
    print(f"loaded embeddings <- {path}")
    return embeddings, word_to_id


def save_checkpoint(model: NegativeSampling, vocab: Vocab, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": model.center_embeddings.weight.detach().cpu(),
            "id_to_word": vocab.id_to_word,
        },
        path,
    )
    print(f"saved embeddings -> {path}")


@torch.no_grad()
def eval_matrix(
    embeddings: torch.Tensor,
    word_to_id: dict[str, int],
    questions,
    title: str,
) -> None:
    for restrict, label in ((None, "full vocab"), (RESTRICT_VOCAB, f"top {RESTRICT_VOCAB:,}")):
        report = evaluate_analogies(
            embeddings,
            word_to_id,
            questions,
            restrict_vocab=restrict,
        )
        print(format_report(report, title=f"{title} ({label})"))
        print()


def train_gensim(corpus_path: Path):
    from gensim.models import Word2Vec
    from gensim.models.word2vec import Text8Corpus

    workers = os.cpu_count() or 1
    print(f"training gensim skip-gram NEG with {workers} workers")
    return Word2Vec(
        sentences=Text8Corpus(str(corpus_path), max_sentence_length=1000),
        vector_size=EMBEDDING_DIM,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        sg=1,
        hs=0,
        negative=NUM_NEGATIVES,
        ns_exponent=0.75,
        sample=SUBSAMPLE_THRESHOLD,
        epochs=EPOCHS,
        alpha=LEARNING_RATE,
        min_alpha=1e-4,
        seed=SEED,
        workers=workers,
        shrink_windows=True,
    )


def eval_gensim(model, questions) -> None:
    keys = list(model.wv.index_to_key)
    word_to_id = {word: index for index, word in enumerate(keys)}
    embeddings = torch.from_numpy(model.wv.vectors.copy())
    eval_matrix(embeddings, word_to_id, questions, title="gensim skip-gram NEG")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip our training and load the saved text8 checkpoint.",
    )
    parser.add_argument("--skip-gensim", action="store_true")
    args = parser.parse_args()

    corpus_path = prepare_text8(DATA_DIR)
    questions_path = prepare_questions_words(DATA_DIR)
    questions = load_questions(questions_path)
    print(f"analogy questions: {len(questions):,}")
    chosen = device()

    if args.eval_only:
        embeddings, word_to_id = load_checkpoint(CHECKPOINT)
        eval_matrix(embeddings.to(chosen), word_to_id, questions, title="ours skip-gram NEG")
    else:
        model, vocab = train_ours(corpus_path, chosen)
        model.eval()
        save_checkpoint(model, vocab, CHECKPOINT)
        eval_matrix(
            model.center_embeddings.weight,
            vocab.word_to_id,
            questions,
            title="ours skip-gram NEG",
        )
        del model
        if chosen.type == "cuda":
            torch.cuda.empty_cache()

    if args.skip_gensim:
        return
    try:
        gensim_model = train_gensim(corpus_path)
    except ImportError:
        print("gensim is not installed; skip the C-compatible reference.")
        print("Install with: pip install gensim")
        return
    eval_gensim(gensim_model, questions)


if __name__ == "__main__":
    main()
