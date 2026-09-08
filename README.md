# nlp-foundation

Word2vec from scratch: skip-gram or CBOW, with hierarchical softmax or
negative sampling.

## Data

`data/text8m1.txt` is the first 1,000,000 words of **text8**, the cleaned
Wikipedia corpus used by the official word2vec demo (`demo-word.sh`). The
papers trained on Google News, which is not public; text8 is the small
public dataset the authors distributed for experiments.

See `data/SOURCE.txt` for download details.

## Train

```bash
python train_word2vec.py
```

That runs skip-gram negative sampling. Pass `architecture="cbow"` into
`negative_sampling()` or `hierarchical_softmax()` for CBOW (mean of context
vectors predicts the center word). Huffman HS does not allocate the NEG
noise table. Frequent-word subsampling is redrawn each epoch on the encoded
stream, then windows are rebuilt inside newline sentences and 1000-token
buffers (as in the original word2vec C code).

## Tests

```bash
python -m unittest discover -s tests
```
