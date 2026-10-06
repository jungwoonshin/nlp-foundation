"""Train Luong et al. (2015) attention NMT on IWSLT'15 English–Vietnamese."""

from __future__ import annotations

from scripts.luong_attention.prepare import prepare_iwslt15
from seq_to_seq.attention.config import LuongConfig
from seq_to_seq.attention.data.dataset import process_parallel, smoke_parallel
from seq_to_seq.attention.data.vocab import load_stanford_vocab
from seq_to_seq.attention.model.nmt import LuongNMT
from seq_to_seq.attention.training.loop import (
    ROOT,
    device,
    fit,
    initialize_parameters,
    log_corpus,
)

DATA_DIR = ROOT / "data"


def luong_attention(*, smoke: bool = False) -> dict[str, float]:
    chosen = device()
    if smoke:
        config = LuongConfig.smoke()
        processed = smoke_parallel(config)
        log_corpus(processed)
        model = LuongNMT(
            processed.src_vocab_size,
            processed.tgt_vocab_size,
            config,
        ).to(chosen)
        initialize_parameters(model, config.init_range, seed=config.seed)
        return fit(
            model,
            processed,
            chosen,
            config,
            eval_processed=processed,
            eval_bleu=True,
        )

    config = LuongConfig()
    config.validate()
    raw = prepare_iwslt15(DATA_DIR)
    src_vocab = load_stanford_vocab(raw / f"vocab.{config.src_lang}")
    tgt_vocab = load_stanford_vocab(raw / f"vocab.{config.tgt_lang}")
    train = process_parallel(
        raw / f"train.{config.src_lang}",
        raw / f"train.{config.tgt_lang}",
        src_vocab,
        tgt_vocab,
        config,
        filter_length=True,
        max_examples=config.max_train_examples,
    )
    dev = process_parallel(
        raw / f"tst2012.{config.src_lang}",
        raw / f"tst2012.{config.tgt_lang}",
        src_vocab,
        tgt_vocab,
        config,
        filter_length=True,
        max_examples=config.max_eval_examples,
    )
    log_corpus(train)
    print(f"dev examples: {len(dev.dataset):,}")
    model = LuongNMT(train.src_vocab_size, train.tgt_vocab_size, config).to(chosen)
    initialize_parameters(model, config.init_range, seed=config.seed)
    return fit(model, train, chosen, config, eval_processed=dev, eval_bleu=True)


if __name__ == "__main__":
    luong_attention(smoke=True)
