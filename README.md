# nlp-foundation

Word2vec from scratch (skip-gram with hierarchical softmax or negative sampling).

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

That runs negative sampling. For Huffman hierarchical softmax, call
`hierarchical_softmax()` in that file (it builds Huffman codes and does
not allocate the NEG noise table).

## Tests

```bash
python -m unittest discover -s tests
```
