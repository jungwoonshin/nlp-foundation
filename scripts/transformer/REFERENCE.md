# Multi30k Transformer-Tiny reference experiment

Reference: Wu et al., ACL 2021, *Good for Misconceived Reasons*.
Author code: https://github.com/LividWo/Revisit-MMT/tree/dc368d0af8d60270b8f4aaa2f3ac58771c551da3
Published text-only English-to-German Test2016 result: **41.02 BLEU**.
This is not a score from Vaswani et al. (2017).

The implementation is aligned in four areas:

| Area | Aligned behavior |
|---|---|
| Model | 4+4 post-LN layers, width 128, FFN 256, 4 heads; attention biases and scaled Xavier Q/K/V initialization; Xavier FFNs; normal embeddings with standard deviation 1/sqrt(128); shared source/target/output embeddings; Fairseq concatenated sinusoids, offset positions, and zero PAD positional row. |
| Data | Authors' released 10k-merge BPE, exact SHA256 checks and 29,000/1,014/1,000 splits; joined 9,712-entry dictionary; source left padding, target right padding/EOS start; stable source/target length ordering and batch-size multiples of 8. |
| Training | 8,000 updates, token batches of 4,096, dropout .3, label smoothing .1, seed 1, warmup 2,000 to .005 from 1e-7 then inverse-square-root decay, Adam (.9,.98) and epsilon 1e-8 in the pinned Fairseq convention, clipping 25 and patience 10. |
| Evaluation | Mean of last 10 complete epoch checkpoints; beam 5, length penalty 1, minimum content length 1, cap 200 plus EOS; remove BPE and score case-sensitive unsmoothed corpus BLEU-4, with escaped reference UNKs. |

The author command uses two GPUs. This runner uses one GPU with two microbatches per update, normalizing by their combined non-PAD token count. This matches effective batch size but not the independent per-rank dropout random streams or DDP reduction order. An additional same-seed source check found all 129 tested embedding and projection parameter tensors identical, so initialization order is not a demonstrated residual difference. PyTorch/CUDA versions also differ. These are residual sources of variation, so 41.02 is a comparison target, not a guaranteed exact result.

The source-loaded parity harness in `.tmp/transformer-multi30k/parity.py` executes unchanged author classes with dependency adapters, copies parameters, and checks encoder/decoder forward results, exact position tables, and Adam updates. This is layer parity, not a successful original-framework training run.

Run from the repository root using a CUDA-enabled Python environment:

```powershell
python prepare.py transformer
python train.py transformer --profile multi30k_tiny --device cuda --output-dir .tmp/transformer-multi30k/run
python eval.py transformer --device cuda --checkpoint .tmp/transformer-multi30k/run --average-last-ten
```

Training refuses to overwrite a nonempty run directory without explicit resume. Checkpoints record source hashes, configuration and dataset provenance. A partial final epoch is not included in the ten-epoch average, matching the author checkpoint selection. Old complete epoch checkpoints beyond the last ten are removed to bound disk usage. No model changes should be made between training and evaluation.

The `--smoke` path uses a dummy model only and must not be compared with the published score.
