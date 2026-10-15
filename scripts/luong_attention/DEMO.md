# Luong toy comparison

## Executed comparison: global/general versus OpenNMT-py

The agreed comparison uses OpenNMT-py, an established independent implementation,
not the authors' MATLAB implementation. `compare_opennmt.py` imports the unchanged
OpenNMT encoder, input-feeding decoder, general attention and embeddings from a
pinned checkout, plus its standard biased linear generator. A shared trainer and
greedy decoder protocol control the experiment. This is a model implementation
comparison, not a benchmark of OpenNMT's default CLI training pipeline.

```powershell
$work = '.tmp/luong-comparison-2026-10-12'
git clone https://github.com/lmthang/nmt.matlab.git "$work/official"
git -C "$work/official" checkout 32f8564e39d4a3d30fa57038e44f4a3dbdc2068c
git clone https://github.com/OpenNMT/OpenNMT-py.git "$work/OpenNMT-py"
git -C "$work/OpenNMT-py" checkout 97111d97551c24857076a4102eabdb468b35cff4
$env:UV_CACHE_DIR = "$PWD/$work/uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$PWD/$work/python"
uv venv "$work/venv" --python 3.11
uv pip install --python "$work/venv/Scripts/python.exe" --index-strategy unsafe-best-match -r scripts/luong_attention/requirements-opennmt.txt
$env:PYTHONDONTWRITEBYTECODE = '1'
& "$work/venv/Scripts/python.exe" scripts/luong_attention/compare_opennmt.py --work-dir $work --include-baseline
& "$work/venv/Scripts/python.exe" scripts/luong_attention/compare_opennmt.py --work-dir $work --include-baseline --verify-only
```

Use an unused work directory for a fresh experiment; existing run directories are
never overwritten. Both repositories must be clean at their pinned revisions.
CUDA is required. Outputs are under `WORK_DIR/global_comparison`: configuration,
code/data hashes, parity checks, initial/final checkpoints, epoch logs, predictions,
and JSON results. `--verify-only` reloads checkpoints and recomputes the scores.

Both models use the official German-English 10,000/100/100 toy corpus, train-only
1,000-word base vocabularies (model source/target sizes 1001/1002), one 100-unit LSTM
layer, 100-dimensional embeddings, input feeding, source reversal, and dropout 0.
All training pairs are retained, source reversal precedes truncation to 50 words,
targets are truncated to 50, and validation/test sequences are untruncated.
Target-length sorting precedes truncation. Each of 10 epochs has 79 batches,
including the 16-example remainder. SGD starts at 1.0 and halves after epoch 5;
gradient norm is limited to 5. Seeds 42 and 43 each get independently sampled
uniform [-0.1,0.1] parameters, mapped identically between implementations, and
identical batch permutations. The final checkpoint is evaluated without test tuning.

The aligned profile sets `rnn_bias=True`, `output_bias=True`, `state_clip=None`,
and `training_convention='opennmt'`: sentence-normalized loss uses actual batch
size and gradient clipping includes embeddings. Existing MATLAB-style defaults
remain available. The optional seed-42 baseline uses that MATLAB profile with
the same initial shared matrices; it is a combined-profile comparison, not an
isolated bias ablation. The general attention matrix is transposed during weight
mapping, since the two projects store equivalent projections differently.

Before training, the runner compares full logits, every parameter gradient, a
clipped SGD update, and greedy outputs for unequal source lengths, target padding,
and one/two layers, on CPU float64 and GPU float32. Perplexity includes EOS and
UNK, excludes PAD, and uses token-average NLL. BLEU is case-sensitive, pretokenized,
unsmoothed corpus BLEU-4 against raw references, with greedy decoding capped at
50 tokens. Tiny-data scores are not the paper's WMT results; two seeds do not
establish statistical significance. Dropout remains zero because this repository's
LuongNMT does not currently implement its nonzero configuration setting.

## Earlier local-attention setup (MATLAB launcher not executed)

The authors' smallest bundled translation demo has 10,000 training, 100 validation,
and 100 test pairs (German to English). This is a small-data comparison, not a
reproduction of the paper's full WMT scores. The earlier `demo.py` and
`official_demo.m` launchers cover local-m/general and local-p/general. They are
retained for reproducibility of preliminary work; MATLAB was unavailable, and
the final agreed comparison above supersedes this plan.

Use Python with CUDA-enabled PyTorch and MATLAB with GPU support for the official
implementation. Scratch data, checkpoints and logs stay under the chosen .tmp directory.

```powershell
git clone https://github.com/lmthang/nmt.matlab.git .tmp/luong-demo/official
git -C .tmp/luong-demo/official checkout 32f8564e39d4a3d30fa57038e44f4a3dbdc2068c
python scripts/luong_attention/demo.py --work-dir .tmp/luong-demo --prepare-only
python scripts/luong_attention/demo.py --work-dir .tmp/luong-demo --label after
matlab -batch "addpath('scripts/luong_attention'); official_demo(fullfile(pwd,'.tmp','luong-demo'))"
```

Both use one 100-unit LSTM layer, 100-dimensional embeddings, no dropout, 1,000-word
base vocabularies, D=10, batch size 128, SGD at 1.0, 10 epochs, seed 42, reversal,
and input feeding. Training retains all pairs and truncates to 50 words after
source reversal; evaluation is untruncated. Target-length sorting precedes
truncation. Vocabulary frequency ties use first occurrence and the same vocabulary
files are supplied to MATLAB. Source/target model vocabulary sizes are 1001/1002.

The official launcher sets epochIter=79 explicitly so both trainers halve the
learning rate after epoch 5 (native epochIter=0 takes a different branch). Both
scale training loss gradients by the configured batch size, including the short
16-example batch. Training logs retain loss per actual sentence.

Compare final-epoch test perplexity (exp of token-average teacher-forced NLL,
including EOS and excluding padding). Do not mix it with the official model.mat
best-validation checkpoint: use modelRecent.mat at iteration 790. The Python
runner also records greedy tokenized BLEU-4 on raw references; the MATLAB launcher
currently collects perplexity and mean NLL only. BLEU is not a cross-framework
comparison until decoding is matched. A shared seed does not imply identical
MATLAB/PyTorch random initializations or batch permutations.

For a before/after check, save the original training loop under the work directory
and pass --baseline-loop PATH --label before. Existing run folders are never
overwritten. Metadata, vocabulary files, code hashes, configuration, logs,
checkpoints and predictions accompany each run. The official source remains
unmodified. The wrappers do not eliminate all framework differences (notably
MATLAB's recurrent backward clipping and random-number streams).
