"""Pinned legacy Fairseq smoothing: epsilon/V across all vocabulary classes."""
from transformer.data.vocab import PAD_ID


def token_loss(logits, target, epsilon=0.):
    """Return summed smoothed loss, unsmoothed NLL, and non-PAD token count."""
    if not 0 <= epsilon < 1:
        raise ValueError("epsilon must be in [0,1)")
    if logits.shape[:-1] != target.shape:
        raise ValueError("logits and target shapes disagree")
    log_probs = logits.log_softmax(-1)
    valid = target.ne(PAD_ID)
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)[valid].sum()
    smooth = -log_probs[valid].sum()
    return (1 - epsilon) * nll + epsilon / logits.size(-1) * smooth, nll, int(valid.sum())
