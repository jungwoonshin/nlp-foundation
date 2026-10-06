"""Usage: python eval.py transformer --checkpoint PATH [--average-last-ten]."""
import argparse
import json
from pathlib import Path
import torch
from scripts.paths import DATA_DIR
from transformer.config import TransformerConfig
from transformer.data.corpus import load_pairs, smoke_pairs
from transformer.data.dataset import process_pairs
from transformer.data.download import download_multi30k
from transformer.data.vocab import Vocab
from transformer.eval.bleu import SCORER
from transformer.eval.metrics import evaluate_perplexity, evaluate_bleu
from transformer.model.nmt import TransformerNMT
from transformer.training.checkpoints import load_checkpoint, average_checkpoints
from transformer.training.smoke import DummyTransformer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--average-last-ten", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    checkpoint = args.checkpoint
    if args.average_last_ten:
        folder = checkpoint if checkpoint.is_dir() else checkpoint.parent
        checkpoint = average_checkpoints(sorted(folder.glob("checkpoint_epoch_*.pt")), folder / "checkpoint_last10_avg.pt")
    payload = load_checkpoint(checkpoint)
    metadata = payload["metadata"]
    config = TransformerConfig(**metadata["config"])
    config.validate()
    vocab = Vocab(**metadata["vocab"])
    dummy = metadata["model_kind"] == "dummy_infrastructure"
    if dummy:
        pairs = smoke_pairs()
        model = DummyTransformer(len(vocab), config.d_model)
    else:
        directory = download_multi30k(args.data_dir)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest != metadata["data_manifest"]:
            raise ValueError("evaluation dataset differs from training provenance")
        pairs = load_pairs(directory / "test.2016.en", directory / "test.2016.de")
        model = TransformerNMT(len(vocab), config)
    model.load_state_dict(payload["model"])
    chosen = torch.device(args.device)
    model.to(chosen)
    processed = process_pairs(pairs, vocab, config.max_positions)
    ppl = evaluate_perplexity(model, processed, chosen, max_tokens=config.max_tokens)
    decoded = evaluate_bleu(model, processed, chosen, beam_size=config.beam_size,
                            max_decode_tokens=config.max_decode_tokens, length_penalty=config.length_penalty,
                            max_tokens=config.max_tokens)
    report = {"kind": metadata["model_kind"], "split": "smoke fixture" if dummy else "test.2016",
              **ppl, "bleu": decoded["bleu"], "scorer": SCORER, "beam_size": config.beam_size,
              "length_penalty": config.length_penalty, "max_decode_tokens": config.max_decode_tokens,
              "reference_bleu": None if dummy else 41.02,
              "notice": "Infrastructure only; not Transformer results." if dummy else "Published baseline comparison requires matched settings."}
    destination = checkpoint.parent
    (destination / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name in ("hypotheses", "references"):
        (destination / f"{name}.de").write_text("\n".join(" ".join(r) for r in decoded[name]) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
