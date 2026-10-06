---
name: Transformer scaffolding
overview: "Scaffold transformer/ for Attention Is All You Need: model math remains signatures only; data, training, evaluation, scripts, and tests are implemented. Use the published Multi30k text-only Transformer-Tiny baseline."
todos:
  - id: scaffold
    content: Create package structure, TransformerConfig, and documented model interfaces
    status: completed
  - id: data
    content: Download reference BPE text, build a shared vocabulary, implement batching and offline smoke data
    status: completed
  - id: model-stubs
    content: Add Transformer component signatures with NotImplementedError
    status: completed
  - id: train-eval
    content: Implement training, checkpoints, decoding, perplexity, and corpus BLEU
    status: completed
  - id: tests
    content: Verify infrastructure using DummyTransformer and reference fixtures
    status: completed
isProject: false
---

# Transformer scaffolding — Attention Is All You Need

## Summary and benchmark

Implement the surrounding pipeline and leave Transformer mathematics for the learner. Use text-only Multi30k English→German: 29,000 training pairs, 1,014 validation pairs, and 1,000 Test2016 pairs. Wu et al. (ACL 2021) report 41.02 BLEU for their 2.6M-parameter Transformer-Tiny. This is a later published baseline, not a Vaswani et al. demo result or a scaffolding acceptance threshold.

Sources: [original architecture](https://arxiv.org/abs/1706.03762), [small-data reference paper](https://aclanthology.org/2021.acl-long.480.pdf), [baseline instructions](https://github.com/LividWo/Revisit-MMT/blob/master/README-baseline.md).

## Package and interfaces

Use the existing `transformer/` directory, with `config.py`, `data/`, `model/`, `training/`, and `eval/` packages. Model files: types, embeddings, positional_encoding, attention, feed_forward, normalization, encoder, decoder, nmt. Implemented data files: download, vocab, corpus, dataset, masks. Training files: loss, schedule, checkpoints, loop. Evaluation files: decoding, bleu, metrics.

Add `scripts/transformer/{prepare,train,eval}/run.py` for the existing root dispatchers:

```bash
python prepare.py transformer
python train.py transformer --smoke
python train.py transformer --profile multi30k_tiny
python eval.py transformer --checkpoint <path>
```

Document public model signatures for token embeddings, sinusoidal positions, scaled dot-product and multi-head attention, ReLU feed-forward layers, layer normalization and residual connections, encoder/decoder layers and stacks, and TransformerNMT. `encode(src, src_lengths)` returns `(B,S,d_model)` memory; `forward(src, src_lengths, tgt_in)` returns `(B,T,V)` logits; per-head attention weights have shape `(B,heads,Q,K)`.

Specify embedding scaling, shared source/target/output weights, post-residual normalization, causal decoder self-attention, and encoder–decoder attention. Model constructors initialize nn.Module then raise NotImplementedError; model math methods also raise. Do not create trainable layers or use an implemented Transformer/attention shortcut. Types are implemented normally.

## Data

Pin Revisit-MMT to `dc368d0af8d60270b8f4aaa2f3ac58771c551da3`. Download `train.en/de`, `valid.en/de`, `test.2016.en/de`, and BPE `code` from `data/multi30k-en-de/`, under gitignored `data/raw/multi30k.en-de/`. Record URLs, commit, SHA256, and split counts in provenance metadata. Use released BPE text directly; no new tokenizer training or visual features.

Build a training-only joined vocabulary matching pinned Fairseq frequency ordering (lexical ties) and padding to a multiple of eight. IDs: BOS=0, PAD=1, EOS=2, UNK=3. Source includes terminal EOS; `tgt_in=EOS+tokens`, `tgt_out=tokens+EOS`. Source padding is on the left, target padding on the right. Do not reuse Luong target-vocabulary conversion. Retain all validation/test pairs; reject empty content and misaligned files. Store original BPE target text for BLEU. Provide an offline hardcoded smoke corpus.

Batch keys: src, src_lengths, tgt_in, tgt_out. Implement causal/key-padding masks with True meaning blocked. Length-grouped batches use a padded max(source,target) token budget of 4096.

## Training and evaluation

Reference profile: encoder/decoder layers 4/4, d_model=128, d_ff=256, heads=4, residual/embedding dropout=.3, attention/activation dropout=0. Adam betas=(.9,.98), epsilon=1e-8, weight decay=0, label smoothing=.1, gradient norm clip=25, seed=1. Linear LR warmup from 1e-7 to .005 over 2000 updates, then inverse-square-root decay. Two microbatches per update approximate the released two-GPU budget. Stop at 8000 updates or after ten epochs without validation-loss improvement. Normalize accumulated loss by the total number of valid target tokens, not microbatch means.

Loss stays outside model; match reference smoothing `(1-epsilon)*NLL + epsilon/V*sum(-log_prob)`, ignoring PAD positions. Perplexity uses unsmoothed token NLL. Save epoch checkpoints with optimizer, scheduler/update counters, RNG states, vocabulary, config, data hashes, and code revision. Resume must restore these states and deterministic batch order. Average the last ten epoch model checkpoints for reference evaluation; fail clearly if insufficient checkpoints.

Implement external autoregressive decoding through model.forward: greedy for smoke; beam=5 for reference, sum log probability normalized by length^1, minimum output length one, maximum 200 content tokens followed by forced EOS. No KV cache in this teaching version. Remove BPE continuation markers before case-sensitive unsmoothed corpus BLEU-4; match the legacy Fairseq scorer's unknown-reference handling. Record scorer and decoding settings.

Smoke mode uses a small trainable DummyTransformer, fixed Adam LR=1e-3, one epoch, batch two, and no smoothing. Full mode attempts the real stub and clearly explains that model implementation is required. Dummy metrics are infrastructure checks, not experimental results.

## Tests and acceptance

Unittest coverage: cached downloads/checksums/alignment/counts; shared vocabulary/UNK/EOS/padding; causal and padding masks; hand-calculated smoothed loss and unsmoothed perplexity; warmup/decay boundaries; token-normalized accumulation; checkpoint round-trip/resume/averaging; scripted greedy/beam EOS behavior; BPE removal and BLEU fixtures; model constructors raise NotImplementedError; offline smoke completes backward passes. Run existing repository tests and git diff --check.

Preserve unrelated working-tree changes and Luong code. No README edits, commit, full Transformer training, or claimed BLEU reproduction. Keep scratch files under `.tmp/`; data and experiment artifacts remain gitignored.
