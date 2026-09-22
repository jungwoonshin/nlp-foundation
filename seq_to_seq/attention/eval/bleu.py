from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n < 1 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def corpus_bleu(
    hypotheses: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
) -> float:
    """Tokenized corpus BLEU-4 with brevity penalty, reported on a 0–100 scale.

    n-gram orders with no hypothesis n-grams are skipped so short identical
    sentences can still score 100. Empty hypotheses score 0.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    if not hypotheses:
        return 0.0

    clipped = [0] * 4
    total = [0] * 4
    hyp_len = 0
    ref_len = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_list = list(hyp)
        ref_list = list(ref)
        hyp_len += len(hyp_list)
        ref_len += len(ref_list)
        for n in range(1, 5):
            hyp_counts = Counter(_ngrams(hyp_list, n))
            ref_counts = Counter(_ngrams(ref_list, n))
            total[n - 1] += sum(hyp_counts.values())
            for gram, count in hyp_counts.items():
                clipped[n - 1] += min(count, ref_counts.get(gram, 0))

    if hyp_len == 0:
        return 0.0

    log_prec = 0.0
    used = 0
    for n in range(4):
        if total[n] == 0:
            continue
        precision = clipped[n] / total[n]
        if precision <= 0.0:
            return 0.0
        log_prec += math.log(precision)
        used += 1
    if used == 0:
        return 0.0

    if hyp_len >= ref_len:
        brevity = 1.0
    else:
        brevity = math.exp(1.0 - ref_len / hyp_len)
    return 100.0 * brevity * math.exp(log_prec / used)
