"""Case-sensitive corpus BLEU-4, no smoothing or effective-order shortcuts."""
from collections import Counter
import math

SCORER = "legacy-fairseq-compatible corpus BLEU-4; tokenized, case-sensitive, unsmoothed"


def remove_bpe(tokens):
    """Remove the reference continuation marker before whitespace tokenization."""
    return " ".join(tokens).replace("@@ ", "").rstrip(" ").removesuffix("@@").split()


def corpus_bleu(hypotheses, references):
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must align")
    matches, counts = [0] * 4, [0] * 4
    hyp_length = ref_length = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_length += len(hyp)
        ref_length += len(ref)
        # The Fairseq scorer makes reference unknown tokens impossible to match.
        ref = [object() if token == "<unk>" else token for token in ref]
        for n in range(1, 5):
            h = Counter(tuple(hyp[i:i + n]) for i in range(max(0, len(hyp) - n + 1)))
            r = Counter(tuple(ref[i:i + n]) for i in range(max(0, len(ref) - n + 1)))
            matches[n - 1] += sum((h & r).values())
            counts[n - 1] += sum(h.values())
    if not hyp_length or any(m == 0 or c == 0 for m, c in zip(matches, counts)):
        return 0.
    bp = min(1., math.exp(1 - ref_length / hyp_length))
    return 100 * bp * math.exp(sum(math.log(m / c) for m, c in zip(matches, counts)) / 4)
