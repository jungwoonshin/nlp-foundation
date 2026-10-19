"""Batched, uncached beam search matching the reference's score and EOS rules.

Encoder memory is reused. Target prefixes are recomputed; no incremental cache.
"""
import torch
from transformer.data.vocab import PAD_ID, EOS_ID


def _prepare_source(model, src, src_lengths, beam_size):
    """Expand source inputs per beam and encode once when supported."""
    memory = source_mask = None
    if hasattr(model, 'encode') and hasattr(model, 'decode') and hasattr(model, 'output_projection'):
        source_mask = src.eq(PAD_ID)[:, None, None, :]
        memory = model.encode(src, source_mask).repeat_interleave(beam_size, dim=0)
        source_mask = source_mask.repeat_interleave(beam_size, dim=0)
    expanded_src = src.repeat_interleave(beam_size, dim=0)
    expanded_lengths = src_lengths.repeat_interleave(beam_size, dim=0)
    return memory, source_mask, expanded_src, expanded_lengths


def _next_token_log_probs(model, prefixes, memory, source_mask, expanded_src, expanded_lengths):
    """Compute next-token log probabilities for every active beam."""
    batch, beam_size, _ = prefixes.shape
    flat = prefixes.reshape(batch * beam_size, -1)
    if memory is None:
        logits = model(expanded_src, expanded_lengths, flat)[:, -1]
    else:
        features = model.decode(flat, memory, source_mask)
        logits = model.output_projection(features[:, -1])
    return logits.float().log_softmax(-1).view(batch, beam_size, -1)


def _apply_token_rules(log_probs, step, max_tokens):
    """Block PAD, require one content token, and force EOS at the limit."""
    log_probs[:, :, PAD_ID] = -float('inf')
    if step == 0:
        log_probs[:, :, EOS_ID] = -float('inf')
    if step == max_tokens:
        eos = log_probs[:, :, EOS_ID].clone()
        log_probs.fill_(-float('inf'))
        log_probs[:, :, EOS_ID] = eos


def _rank_candidates(log_probs, cumulative, beam_size):
    """Rank up to twice the beam size across all parent/token pairs."""
    scores = (log_probs + cumulative[:, :, None]).flatten(1)
    top_scores, top_indices = scores.topk(min(2 * beam_size, scores.size(1)), dim=1)
    return top_scores.tolist(), top_indices.tolist()


def _advance_sentence(prefixes, scores, indices, finished, done, *,
                      vocab_size, beam_size, step, length_penalty):
    """Finalize EOS candidates and select the next live beams for one sentence."""
    live, live_scores = [], []
    if not done:
        for rank, (score, index) in enumerate(zip(scores, indices)):
            if score == -float('inf'):
                continue
            parent, token = divmod(index, vocab_size)
            sequence = prefixes[parent] + [token]
            if token == EOS_ID:
                if rank < beam_size:
                    finished.append((sequence[1:], score / ((step + 1) ** length_penalty)))
            elif len(live) < beam_size:
                live.append(sequence)
                live_scores.append(score)
        finished.sort(key=lambda item: item[1], reverse=True)
        finished = finished[:beam_size]
        done = len(finished) >= beam_size or not live
    while len(live) < beam_size:
        live.append([EOS_ID] * (step + 2))
        live_scores.append(-float('inf'))
    return live, live_scores, finished, done


def _pad_best_sequences(finished, src):
    """Select the best completed sequence per sentence and right-pad the batch."""
    if any(not row for row in finished):
        raise RuntimeError("decoder could not produce EOS within its configured limit")
    best = [row[0][0] for row in finished]
    result = src.new_full((len(finished), max(map(len, best))), PAD_ID)
    for b, values in enumerate(best):
        result[b, :len(values)] = src.new_tensor(values)
    return result


@torch.no_grad()
def generate(model, src, src_lengths, *, beam_size=5, max_tokens=200, length_penalty=1.):
    """Generate translations by preparing sources, expanding beams and selecting results."""
    if beam_size < 1 or max_tokens < 1 or length_penalty < 0:
        raise ValueError("invalid decoding limits")
    model.eval()
    batch = src.size(0)
    prefixes = src.new_full((batch, beam_size, 1), EOS_ID)
    cumulative = torch.full((batch, beam_size), -float('inf'), device=src.device)
    cumulative[:, 0] = 0.
    finished = [[] for _ in range(batch)]
    done = [False] * batch
    memory, source_mask, expanded_src, expanded_lengths = _prepare_source(
        model, src, src_lengths, beam_size
    )
    for step in range(max_tokens + 1):
        log_probs = _next_token_log_probs(
            model, prefixes, memory, source_mask, expanded_src, expanded_lengths
        )
        _apply_token_rules(log_probs, step, max_tokens)
        candidate_scores, candidate_ids = _rank_candidates(log_probs, cumulative, beam_size)
        old_prefixes = prefixes.tolist()
        next_prefixes, next_scores = [], []
        for b in range(batch):
            live, live_scores, finished[b], done[b] = _advance_sentence(
                old_prefixes[b], candidate_scores[b], candidate_ids[b], finished[b], done[b],
                vocab_size=log_probs.size(-1), beam_size=beam_size,
                step=step, length_penalty=length_penalty,
            )
            next_prefixes.append(live)
            next_scores.append(live_scores)
        if all(done):
            break
        prefixes = src.new_tensor(next_prefixes)
        cumulative = torch.tensor(next_scores, device=src.device)
    return _pad_best_sequences(finished, src)
