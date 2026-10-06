"""Usage: python train.py transformer --smoke | --profile multi30k_tiny."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import torch
from scripts.paths import DATA_DIR, REPO_ROOT
from transformer.config import TransformerConfig
from transformer.data.corpus import load_pairs, smoke_pairs
from transformer.data.dataset import process_pairs
from transformer.data.download import download_multi30k
from transformer.data.vocab import build_vocab
from transformer.model.nmt import TransformerNMT
from transformer.training.loop import fit
from transformer.training.smoke import DummyTransformer


def code_provenance():
    roots = (REPO_ROOT / "transformer", REPO_ROOT / "scripts/transformer")
    hashes = {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for root in roots for path in sorted(root.rglob("*.py"))}
    return {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            "code_sha256": hashes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile", choices=["multi30k_tiny"], default="multi30k_tiny")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".tmp/transformer/runs")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    config = TransformerConfig.smoke() if args.smoke else TransformerConfig()
    config.validate()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    chosen = torch.device(args.device)
    provenance = code_provenance()
    if args.smoke:
        pairs = smoke_pairs()
        train_pairs, valid_pairs = pairs, pairs
        provenance["data_manifest"] = {"smoke": True, "evaluation_split": "same offline fixture; infrastructure only"}
    else:
        # Fail clearly before a potentially unnecessary dataset download.
        try:
            TransformerNMT(4, config)
        except NotImplementedError as error:
            parser.exit(2, f"{error}. Use --smoke for infrastructure verification.\n")
        directory = download_multi30k(args.data_dir)
        train_pairs = load_pairs(directory / "train.en", directory / "train.de")
        valid_pairs = load_pairs(directory / "valid.en", directory / "valid.de")
        provenance["data_manifest"] = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    vocab = build_vocab(train_pairs)
    train, valid = (process_pairs(p, vocab, config.max_positions) for p in (train_pairs, valid_pairs))
    model = (DummyTransformer(len(vocab), config.d_model) if args.smoke else TransformerNMT(len(vocab), config))
    metadata = {**provenance, "config": asdict(config), "vocab": vocab.to_dict(),
                "model_kind": "dummy_infrastructure" if args.smoke else "transformer",
                "reference_bleu": None if args.smoke else 41.02}
    state = fit(model, train, valid, config, chosen, output_dir=args.output_dir, metadata=metadata, resume=args.resume)
    report = {"kind": metadata["model_kind"], "training_state": state,
              "notice": "Dummy metrics verify infrastructure only; not Transformer results." if args.smoke else ""}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training-results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
