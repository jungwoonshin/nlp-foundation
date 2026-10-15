"""Controlled GPU model comparison, using unchanged OpenNMT-py model modules.

This intentionally shares data, training and scoring code; it is not a comparison
of the two projects' default CLI pipelines. See DEMO.md for pinned dependencies.
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
from torch import nn
from torch.nn import functional as F
from seq_to_seq.attention.config import LuongConfig
from seq_to_seq.attention.data.corpus import load_parallel_lines
from seq_to_seq.attention.data.dataset import process_pairs
from seq_to_seq.attention.data.vocab import Vocab
from seq_to_seq.attention.eval.metrics import evaluate_perplexity, ids_to_tokens
from seq_to_seq.attention.model.nmt import LuongNMT
from seq_to_seq.attention.training.loop import fit, initialize_parameters, _clip_gradients

OFFICIAL_SHA = '32f8564e39d4a3d30fa57038e44f4a3dbdc2068c'
OPENNMT_SHA = '97111d97551c24857076a4102eabdb468b35cff4'


class OpenNMTAdapter(nn.Module):
    """Only adapts tensor framing and greedy decoding to the repo interface."""

    def __init__(self, src_size, tgt_size, config):
        super().__init__()
        from onmt.encoders.rnn_encoder import RNNEncoder
        from onmt.decoders.decoder import InputFeedRNNDecoder
        from onmt.modules.embeddings import Embeddings
        self.encoder = RNNEncoder('LSTM', False, config.num_layers, config.hidden_size,
            dropout=0., embeddings=Embeddings(config.embed_dim, src_size, 0))
        self.decoder = InputFeedRNNDecoder('LSTM', False, config.num_layers,
            config.hidden_size, attn_type='general', dropout=0.,
            embeddings=Embeddings(config.embed_dim, tgt_size, 0))
        # Same generator construction as onmt.model_builder.build_base_model.
        self.generator = nn.Linear(config.hidden_size, tgt_size)

    def encode(self, src, lengths):
        enc, state, _ = self.encoder(src.unsqueeze(-1), lengths)
        self.decoder.init_state(src.unsqueeze(-1), enc, state)
        return enc

    def forward(self, src, lengths, tgt_in):
        enc = self.encode(src, lengths)
        out, _ = self.decoder(tgt_in.unsqueeze(-1), enc, src_len=lengths)
        return self.generator(out)

    @torch.no_grad()
    def generate(self, src, lengths, max_len):
        self.eval()
        enc = self.encode(src, lengths)
        prev = src.new_full((src.size(0),), 1)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        steps = []
        for _ in range(max_len):
            out, _ = self.decoder(prev[:, None, None], enc, src_len=lengths)
            token = self.generator(out[:, 0]).argmax(-1)
            token = torch.where(finished, torch.zeros_like(token), token)
            finished |= token == 2
            steps.append(token)
            prev = token
            if bool(finished.all()):
                break
        return torch.stack(steps, 1)


def parameter_pairs(ours, reference):
    """Map equivalent parameters; Luong's general matrix is transposed in ONMT."""
    pairs = [
        ('source_embedding', ours.encoder_embed.weight, reference.encoder.embeddings.word_lut.weight, False),
        ('target_embedding', ours.decoder_embed.weight, reference.decoder.embeddings.word_lut.weight, False),
        ('general', ours.attention.weight_a.weight, reference.decoder.attn.linear_in.weight, True),
        ('attentional_projection', ours.w_c.weight, reference.decoder.attn.linear_out.weight, False),
        ('output', ours.out.weight, reference.generator.weight, False),
    ]
    if ours.out.bias is not None:
        pairs.append(('output_bias', ours.out.bias, reference.generator.bias, False))
    for i in range(ours.num_layers):
        for side in ('encoder', 'decoder'):
            own = getattr(ours, side).cells[i]
            ref = reference.encoder.rnn if side == 'encoder' else reference.decoder.rnn.layers[i]
            for prefix in ('ih', 'hh'):
                for kind in ('weight', 'bias'):
                    param = getattr(getattr(own, prefix), kind)
                    if param is not None:
                        suffix = f'_l{i}' if side == 'encoder' else ''
                        pairs.append((f'{side}.{i}.{prefix}.{kind}', param,
                                      getattr(ref, f'{kind}_{prefix}{suffix}'), False))
    return pairs


@torch.no_grad()
def copy_parameters(ours, reference):
    pairs = parameter_pairs(ours, reference)
    assert sum(p.numel() for _, p, _, _ in pairs) == sum(p.numel() for p in ours.parameters())
    for _, own, ref, transpose in pairs:
        own.copy_(ref.T if transpose else ref)


def check_parity(config, device, dtype):
    """Full logits, gradients, clipped SGD update and greedy parity, mixed lengths."""
    torch.manual_seed(71)
    reference = OpenNMTAdapter(19, 23, config).to(device=device, dtype=dtype)
    initialize_parameters(reference, seed=71)
    ours = LuongNMT(19, 23, config).to(device=device, dtype=dtype)
    copy_parameters(ours, reference)
    src = torch.tensor([[4, 7, 6, 5], [8, 0, 0, 0], [6, 4, 9, 0]], device=device)
    lengths = torch.tensor([4, 1, 3], device=device)
    tgt = torch.tensor([[1, 6, 5], [1, 9, 0], [1, 4, 7]], device=device)
    labels = torch.tensor([[6, 5, 2], [9, 2, 0], [4, 7, 2]], device=device)
    outputs = [model(src, lengths, tgt) for model in (ours, reference)]
    atol = 1e-10 if dtype == torch.float64 else 2e-6
    torch.testing.assert_close(*outputs, atol=atol, rtol=atol)
    for logits in outputs:
        (F.cross_entropy(logits.reshape(-1, 23), labels.reshape(-1),
                         ignore_index=0, reduction='sum') / 3).backward()
    grad_error = 0.
    for _, own, ref, transpose in parameter_pairs(ours, reference):
        expected = ref.grad.T if transpose else ref.grad
        torch.testing.assert_close(own.grad, expected, atol=atol, rtol=atol)
        grad_error = max(grad_error, (own.grad - expected).abs().max().item())
    for model in (ours, reference):
        _clip_gradients(model, .05, convention='opennmt')
        torch.optim.SGD(model.parameters(), lr=.1).step()
    update_error = 0.
    for _, own, ref, transpose in parameter_pairs(ours, reference):
        expected = ref.T if transpose else ref
        torch.testing.assert_close(own, expected, atol=atol, rtol=atol)
        update_error = max(update_error, (own - expected).abs().max().item())
    torch.testing.assert_close(ours.generate(src, lengths, 8), reference.generate(src, lengths, 8))
    return dict(device=str(device), dtype=str(dtype), layers=config.num_layers,
                max_logit_error=(outputs[0]-outputs[1]).abs().max().item(),
                max_gradient_error=grad_error, max_update_error=update_error, passed=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--include-baseline', action='store_true')
    parser.add_argument('--parity-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    base = args.work_dir.resolve()
    for folder, expected in [('official', OFFICIAL_SHA), ('OpenNMT-py', OPENNMT_SHA)]:
        actual = subprocess.check_output(['git', '-C', str(base/folder), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            raise RuntimeError(f'Unexpected {folder} revision: {actual}')
        if subprocess.check_output(['git', '-C', str(base/folder), 'status', '--porcelain'], text=True).strip():
            raise RuntimeError(f'{folder} checkout is modified')
    sys.path.insert(0, str(base/'OpenNMT-py'))
    import onmt
    from sacrebleu.metrics import BLEU
    assert Path(onmt.__file__).resolve().is_relative_to(base/'OpenNMT-py')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; refusing CPU fallback')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    config = LuongConfig(attention='global', score='general', src_lang='de', tgt_lang='en',
        num_layers=1, hidden_size=100, embed_dim=100, batch_size=128, epochs=10,
        dropout=0., seed=42, rnn_bias=True, output_bias=True, state_clip=None,
        training_convention='opennmt')
    result_dir = base/'global_comparison'
    result_dir.mkdir(exist_ok=True)
    parity = [check_parity(replace(config, hidden_size=7, embed_dim=5, num_layers=layers), dev, dtype)
              for layers in (1, 2) for dev, dtype in [('cpu', torch.float64), ('cuda', torch.float32)]]
    (result_dir/'parity.json').write_text(json.dumps(parity, indent=2), encoding='utf-8')
    print('PARITY', json.dumps(parity), flush=True)
    if args.parity_only:
        return
    raw = base/'official'/'data'
    all_pairs = [load_parallel_lines(raw/f'{split}.de', raw/f'{split}.en')
                 for split in ('train.10k', 'valid.100', 'test.100')]
    assert [len(p) for p in all_pairs] == [10000, 100, 100]
    specials = ['<pad>', '<s>', '</s>', '<unk>']
    vocabs = [Vocab(specials + [t for t, _ in Counter(t for p in all_pairs[0] for t in p[side]).most_common(997)
                              if t not in specials]) for side in (0, 1)]
    all_pairs[0] = [(s[-50:], t[:50]) for s, t in sorted(all_pairs[0], key=lambda p: len(p[1]))]
    train, valid, test = [process_pairs(p, *vocabs, config, filter_length=False) for p in all_pairs]
    for side, vocab in [('source', train.src_vocab), ('target', train.tgt_vocab)]:
        (result_dir/f'{side}-vocab.json').write_text(json.dumps(vocab.id_to_token, ensure_ascii=False), encoding='utf-8')
    code_files = sorted((ROOT/'seq_to_seq'/'attention').rglob('*.py')) + [Path(__file__)]
    code_hashes = {str(p.relative_to(ROOT)): sha(p) for p in code_files}
    metadata = dict(config=asdict(config), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        cuda=torch.version.cuda, python=sys.version, official_sha=OFFICIAL_SHA, opennmt_sha=OPENNMT_SHA,
        onmt_source=onmt.__file__, data_sha256={p.name: sha(p) for p in raw.iterdir() if p.name in
          [f'{s}.{l}' for s in ('train.10k', 'valid.100', 'test.100') for l in ('de', 'en')]},
        code_sha256=code_hashes, vocab_sizes=[train.src_vocab_size, train.tgt_vocab_size],
        protocol='Shared trainer, sentence normalization, full gradient norm; final epoch; identical mapped initialization and batch RNG per seed')
    if not args.verify_only:
        (result_dir/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    bleu = BLEU(tokenize='none', smooth_method='none', effective_order=False)
    references = [' '.join(p[1]) for p in all_pairs[2]]
    all_results = []
    for seed in args.seeds:
        cfg = replace(config, seed=seed)
        reference = OpenNMTAdapter(train.src_vocab_size, train.tgt_vocab_size, cfg).to(device)
        initialize_parameters(reference, seed=seed)
        # Freeze one canonical initialization for all implementations within this seed.
        initial = {k: v.detach().clone() for k, v in reference.state_dict().items()}
        kinds = ['opennmt', 'ours_aligned']
        if args.include_baseline and seed == args.seeds[0]:
            kinds.insert(0, 'ours_matlab_profile')
        for kind in kinds:
            folder = result_dir/f'{kind}_seed{seed}'
            baseline = kind == 'ours_matlab_profile'
            run_cfg = replace(cfg, rnn_bias=False, output_bias=False, state_clip=50.,
                              training_convention='matlab') if baseline else cfg
            reference.load_state_dict(initial)
            if kind == 'opennmt':
                model = reference
            else:
                model = LuongNMT(train.src_vocab_size, train.tgt_vocab_size, run_cfg).to(device)
                copy_parameters(model, reference)
            if args.verify_only:
                model.load_state_dict(torch.load(folder/'model.pt', map_location=device, weights_only=True))
                saved = json.loads((folder/'results.json').read_text())
            else:
                folder.mkdir(exist_ok=False)
                (folder/'config.json').write_text(json.dumps(asdict(run_cfg), indent=2), encoding='utf-8')
                torch.save(model.state_dict(), folder/'initial.pt')
                torch.manual_seed(seed + 10000)
                start = time.perf_counter()
                print(f'START {kind} seed={seed}', flush=True)
                with (folder/'training.log').open('w', encoding='utf-8', buffering=1) as log:
                    class Tee:
                        def write(self, value):
                            sys.__stdout__.write(value)
                            log.write(value)
                        def flush(self):
                            sys.__stdout__.flush()
                            log.flush()
                    sys.stdout = Tee()
                    try:
                        fit(model, train, device, run_cfg, eval_processed=valid)
                    finally:
                        sys.stdout = sys.__stdout__
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                torch.save(model.state_dict(), folder/'model.pt')
            model.eval()
            ppl = evaluate_perplexity(model, test, device, batch_size=128)
            valid_ppl = evaluate_perplexity(model, valid, device, batch_size=128)
            hypotheses = []
            for batch in test.dataloader(128, shuffle=False):
                pred = model.generate(batch['src'].to(device), batch['src_lengths'].to(device), 50)
                hypotheses.extend(' '.join(ids_to_tokens(row, test.tgt_vocab)) for row in pred)
            score = bleu.corpus_score(hypotheses, [references])
            result = dict(implementation=kind, seed=seed, test_perplexity=ppl, valid_perplexity=valid_ppl,
                test_bleu=score.score, bleu_signature=str(bleu.get_signature()), bleu_counts=score.counts,
                bleu_totals=score.totals, hypothesis_words=score.sys_len, reference_words=score.ref_len,
                parameters=sum(p.numel() for p in model.parameters()),
                train_seconds=saved['train_seconds'] if args.verify_only else seconds, steps=790)
            if args.verify_only:
                assert abs(result['test_perplexity'] - saved['test_perplexity']) < 1e-5
                assert result['test_bleu'] == saved['test_bleu']
                assert hypotheses == (folder/'hypotheses.en').read_text(encoding='utf-8').splitlines()
            else:
                (folder/'hypotheses.en').write_text('\n'.join(hypotheses)+'\n', encoding='utf-8')
                (folder/'references.en').write_text('\n'.join(references)+'\n', encoding='utf-8')
                (folder/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            all_results.append(result)
            print('VERIFIED' if args.verify_only else 'COMPLETE', json.dumps(result), flush=True)
            if kind != 'opennmt':
                del model
    assert all(sha(ROOT/p) == digest for p, digest in code_hashes.items()), 'Code changed during run'
    name = 'verification.json' if args.verify_only else 'comparison.json'
    (result_dir/name).write_text(json.dumps(all_results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
