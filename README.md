# nlp-foundation

Word2vec from scratch.

## Data

`data/text8m1.txt` is the first 1,000,000 words of **text8**, the cleaned
Wikipedia corpus used by the official word2vec demo (`demo-word.sh`). The
papers trained on Google News, which is not public; text8 is the small
public dataset the authors distributed for experiments.

See `data/SOURCE.txt` for download details. Run `python processor.py` to
build a PyTorch skip-gram dataset (data processing only).
