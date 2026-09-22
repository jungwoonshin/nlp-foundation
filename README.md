# nlp-foundation

From-scratch NLP implementations for learning and for checking against
published baselines. The repo includes word embeddings (word2vec, FastText
subwords), text classification (FastText, TextCNN), encoder–decoder RNNs, and
Luong attention NMT.

## Usage

Every model uses the same root entry points; implementations live under
`scripts/<model>/prepare|train|eval/`:

```bash
python prepare.py <model>
python train.py <model>
python eval.py <model>
```

Run a script with no arguments (or `--help`) to list models available for
that stage. `python prepare.py all` downloads or verifies every dataset.

| Model | prepare | train | eval | Task / dataset |
| --- | --- | --- | --- | --- |
| `word2vec` | yes | yes | yes | Skip-gram or CBOW on text8 |
| `subword` | yes | — | yes | FastText SISG on text8; Rare Words eval |
| `fasttext` | yes | yes | yes | Supervised bag-of-tricks on AG News |
| `textcnn` | yes | yes | yes | Kim (2014) CNN on AG News |
| `simple_rnn` | — | yes | — | 1-layer RNN seq2seq (reverse task) |
| `stacked_rnn` | — | yes | — | Stacked RNN seq2seq (reverse task) |
| `stacked_lstm` | — | yes | — | Stacked LSTM seq2seq (reverse task) |
| `luong_attention` | yes | yes | yes | Luong et al. NMT on IWSLT'15 En–Vi |

Seq2seq foundation models train on a synthetic reverse-sequence task (no
download). Luong attention model code is scaffolded; training and eval use
`DummyNMT` until the full architecture is implemented.

## Data

`data/text8m1.txt` is the first 1,000,000 words of **text8**, the cleaned
Wikipedia corpus used by the official word2vec demo (`demo-word.sh`). The
papers trained on Google News, which is not public; text8 is the small
public dataset the authors distributed for experiments.

See `data/SOURCE.txt` for download details on text8, AG News, the Stanford
Rare Words set, and IWSLT'15 English–Vietnamese.

## Word2vec

```bash
python train.py word2vec
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
