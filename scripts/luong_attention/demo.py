"""GPU comparison on the pinned official 10k/100/100 Luong toy corpus.

Clone lmthang/nmt.matlab into WORK_DIR/official first. --prepare-only writes
shared integer data/vocabularies for official_demo.m without training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from lstm_attention.attention.config import LuongConfig
from lstm_attention.attention.data.corpus import load_parallel_lines
from lstm_attention.attention.data.dataset import process_pairs
from lstm_attention.attention.data.vocab import Vocab
from lstm_attention.attention.eval.metrics import evaluate_bleu, evaluate_perplexity, ids_to_tokens
from lstm_attention.attention.training.loop import fit, initialize_parameters
from lstm_attention.attention.model.nmt import LuongNMT

parser = argparse.ArgumentParser()
parser.add_argument('--kinds', nargs='+', choices=('local_m', 'local_p'), default=['local_m', 'local_p'])
parser.add_argument('--work-dir', type=Path, required=True)
parser.add_argument('--prepare-only', action='store_true')
parser.add_argument('--threads', type=int, default=1)
parser.add_argument('--baseline-loop', type=Path)
parser.add_argument('--label', default='after')
args = parser.parse_args()
torch.set_num_threads(args.threads)
base = args.work_dir.resolve()
raw = base / 'official' / 'data'
official_commit = subprocess.check_output(['git', '-C', str(base / 'official'), 'rev-parse', 'HEAD'], text=True).strip()
if official_commit != '32f8564e39d4a3d30fa57038e44f4a3dbdc2068c':
    raise RuntimeError('Unexpected official commit; audit before comparing')
train_pairs = load_parallel_lines(raw / 'train.10k.de', raw / 'train.10k.en')
valid_pairs = load_parallel_lines(raw / 'valid.100.de', raw / 'valid.100.en')
test_pairs = load_parallel_lines(raw / 'test.100.de', raw / 'test.100.en')
raw_train_pairs = train_pairs
# Match official sorting BEFORE truncation, including ties above length 50.
train_pairs = sorted(train_pairs, key=lambda pair: len(pair[1]))
# Official source reversal precedes truncation: retain the last 50 source words.
train_pairs = [(s[-50:], t[:50]) for s, t in train_pairs]
specials = ['<pad>', '<s>', '</s>', '<unk>']
vocabs = [Vocab(specials + [t for t, _ in Counter(
    t for pair in raw_train_pairs for t in pair[side]
).most_common(997) if t not in specials]) for side in (0, 1)]
# Both frameworks consume the same base vocabulary (997 words plus 3 specials).
prepared = base / 'prepared'
prepared.mkdir(exist_ok=True)
manifest = {'official_commit': official_commit, 'files': {}, 'splits': {}}
for lang, vocab in zip(('de', 'en'), vocabs):
    official_words = ['<unk>', '<s>', '</s>', *vocab.id_to_token[4:]]
    assert len(official_words) == 1000 and len(set(official_words)) == 1000
    mapping = {word: i for i, word in enumerate(official_words)}
    (prepared / f'train.10k.{lang}.vocab.1000').write_text('\n'.join(official_words) + '\n', encoding='utf-8')
    for split, count in (('train.10k', 10000), ('valid.100', 100), ('test.100', 100)):
        path = raw / f'{split}.{lang}'
        lines = path.read_text(encoding='utf-8').splitlines()
        assert len(lines) == count
        manifest['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['splits'][split] = count
        rows = [[str(mapping.get(word, 0)) for word in line.split()] for line in lines]
        (prepared / path.name).write_text('\n'.join(' '.join(row) for row in rows) + '\n', encoding='utf-8')
        (prepared / (path.name + '.reversed')).write_text('\n'.join(' '.join(reversed(row)) for row in rows) + '\n', encoding='utf-8')
manifest['vocabulary_policy'] = 'train only; frequency descending; first occurrence breaks ties; shared with MATLAB'
(base / 'dataset-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
if args.prepare_only:
    print(json.dumps(manifest, indent=2))
    raise SystemExit(0)
tracked = subprocess.check_output(
    ['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT, text=True
).splitlines()
code_hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
               for p in tracked if p.endswith('.py') and (ROOT / p).is_file()}
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
patch = subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT)
(base / (args.label + '-working-tree.patch')).write_bytes(patch)
patch_sha256 = hashlib.sha256(patch).hexdigest()
if not torch.cuda.is_available():
    raise RuntimeError('GPU is required; refusing CPU fallback')
device = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if args.baseline_loop:
    import importlib.util
    spec = importlib.util.spec_from_file_location('baseline_loop', args.baseline_loop)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    fit = baseline.fit

for kind in args.kinds:
    folder = base / 'runs' / (args.label + '_' + kind)
    if folder.exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {folder}')
    folder.mkdir(parents=True)
    log_file = (folder / 'training.log').open('w', encoding='utf-8', buffering=1)
    class Tee:
        def write(self, text):
            sys.__stdout__.write(text)
            log_file.write(text)
        def flush(self):
            sys.__stdout__.flush()
            log_file.flush()
    sys.stdout = Tee()
    torch.manual_seed(42)
    config = replace(LuongConfig(), attention=kind.removesuffix('_before'), score='general', reverse_source=True, src_lang='de',
                     tgt_lang='en', num_layers=1, hidden_size=100, embed_dim=100,
                     batch_size=128, epochs=10, optimizer='sgd', learning_rate=1.,
                     lr_decay_start=5, dropout=0., seed=42)
    train = process_pairs(train_pairs, *vocabs, config, filter_length=False)
    valid = process_pairs(valid_pairs, *vocabs, config, filter_length=False)
    test = process_pairs(test_pairs, *vocabs, config, filter_length=False)
    metadata = {
        'head': head, 'official_commit': '32f8564e39d4a3d30fa57038e44f4a3dbdc2068c',
        'config': asdict(config), 'device': str(device), 'threads': args.threads,
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'iteration': args.label, 'train_pairs_raw': len(train_pairs),
        'train_pairs_retained': len(train.dataset), 'valid_pairs': len(valid.dataset),
        'test_pairs': len(test.dataset), 'vocab_sizes': [train.src_vocab_size, train.tgt_vocab_size],
        'working_tree_patch_sha256': patch_sha256,
        'training_truncation': 'source reversed before truncation to 50; target truncated to 50; no sentence filtering',
        'parameters': None, 'checkpoint_policy': 'final epoch 10',
        'batch_sorting': 'original target length before truncation; shuffle batches each epoch',
        'trainer_sha256': hashlib.sha256((args.baseline_loop or ROOT / 'lstm_attention/attention/training/loop.py').read_bytes()).hexdigest(), 'tracked_code_sha256_before': code_hashes,
        'source': 'authors official demo; not the full-scale paper benchmark',
    }
    for name, vocab in zip(('source', 'target'), (train.src_vocab, train.tgt_vocab)):
        (folder / f'{name}-vocab.json').write_text(json.dumps(vocab.id_to_token, ensure_ascii=False), encoding='utf-8')
    model = LuongNMT(train.src_vocab_size, train.tgt_vocab_size, config).to(device)
    initialize_parameters(model, config.init_range, seed=config.seed)
    metadata['parameters'] = sum(p.numel() for p in model.parameters())
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    counter = [0]
    start = time.perf_counter()
    def progress(module, inputs, output):
        if module.training:
            counter[0] += 1
            if counter[0] % 10 == 0:
                print(f'{kind}: training forward batch {counter[0]}, elapsed={time.perf_counter()-start:.1f}s', flush=True)
    handle = model.register_forward_hook(progress)
    print(f'START {kind}: {json.dumps(metadata["config"])}', flush=True)
    trainer = fit
    valid_metrics = trainer(model, train, device, config, eval_processed=valid, eval_bleu=False)
    handle.remove()
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    torch.save(model.state_dict(), folder / 'model.pt')
    test_metrics = {
        'perplexity': evaluate_perplexity(model, test, device, batch_size=128),
        'bleu': evaluate_bleu(model, test, device, batch_size=128, max_len=50),
    }
    hypotheses = []
    model.eval()
    with torch.no_grad():
        for batch in test.dataloader(128, shuffle=False):
            prediction = model.generate(batch['src'].to(device), batch['src_lengths'].to(device), max_len=50)
            hypotheses.extend(' '.join(ids_to_tokens(row, test.tgt_vocab)) for row in prediction)
    (folder / 'hypotheses.en').write_text('\n'.join(hypotheses) + '\n', encoding='utf-8')
    (folder / 'references.en').write_text('\n'.join(' '.join(pair[1]) for pair in test_pairs) + '\n', encoding='utf-8')
    results = {'kind': kind, 'score': 'general', 'valid_metrics': valid_metrics,
               'test_metrics': test_metrics, 'train_seconds': train_seconds,
               'total_seconds': time.perf_counter()-start, 'training_batches': counter[0]}
    unchanged = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == value
                    for p, value in code_hashes.items())
    results['tracked_python_code_unchanged'] = unchanged
    (folder / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('COMPLETE', json.dumps(results), flush=True)
    sys.stdout = sys.__stdout__
    log_file.close()
