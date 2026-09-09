






from pathlib import Path
from train_word2vec import process
from word2vec.negative_sampling import SubwordNegativeSampling
from word2vec.pipeline import ProcessedCorpus
from word2vec.config import ProcessingConfig
from train_word2vec import _log_corpus, _device, _fit
from train_word2vec import _neg_loss
from functools import partial

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "data" / "text8m1.txt"
EMBEDDING_DIM = 24
BATCH_SIZE = 256
LEARNING_RATE = 0.025
EPOCHS = 200    


def subword_negative_sampling(architecture: str = "skipgram") -> None:
    if architecture != "skipgram":
        raise ValueError("Subword NEG only supports skipgram; CBOW bags are not encoded yet.")
    processed = process(
        build_huffman=False,
        build_negative_sampler=True,
        architecture=architecture,
    )
    _log_corpus(processed)
    device = _device()
    model = SubwordNegativeSampling(
        embedding_dim=EMBEDDING_DIM,
        vocab_size=len(processed.vocab),
        vocab=processed.vocab,
    ).to(device)
    _fit(
        model,
        processed,
        device,
        partial(_neg_loss, architecture=architecture),
        with_negatives=True,
    )

if __name__ == "__main__":
    subword_negative_sampling()
    # hierarchical_softmax()
