"""Uncached autoregressive greedy/beam decoding through the forward contract."""
import torch
from transformer.data.vocab import BOS_ID, PAD_ID, EOS_ID


@torch.no_grad()
def generate(model, src, src_lengths, *, beam_size=5, max_tokens=200, length_penalty=1.):
    """Return (B,L) ids excluding decoder-start EOS, padded after generated EOS.

    Use top-2*beam candidates, length-normalized finished scores, no sampling,
    no repeated-ngram blocking. Reference limit: up to max_tokens content
    steps plus one forced EOS step. Minimum content length is one.
    """
    if beam_size < 1 or max_tokens < 1 or length_penalty < 0:
        raise ValueError("invalid decoding limits")
    model.eval()
    outputs = []
    for row in range(src.size(0)):
        active, finished = [([EOS_ID], 0.)], []
        for step in range(max_tokens + 1):
            prefixes = torch.tensor([tokens for tokens, _ in active], device=src.device)
            logits = model(src[row:row + 1].expand(len(active), -1),
                           src_lengths[row:row + 1].expand(len(active)), prefixes)[:, -1]
            scores = logits.log_softmax(-1)
            scores[:, PAD_ID] = -float("inf")
            scores[:, BOS_ID] = -float("inf")
            if step == 0:
                scores[:, EOS_ID] = -float("inf")
            if step == max_tokens:
                eos_scores = scores[:, EOS_ID].clone()
                scores.fill_(-float("inf"))
                scores[:, EOS_ID] = eos_scores
            scores += scores.new_tensor([s for _, s in active])[:, None]
            top_scores, top_indices = scores.flatten().topk(min(2 * beam_size, scores.numel()))
            next_active = []
            for rank, (score, index) in enumerate(zip(top_scores.tolist(), top_indices.tolist())):
                if score == -float("inf"):
                    continue
                parent, token = divmod(index, scores.size(1))
                tokens = [*active[parent][0], token]
                if token == EOS_ID:
                    if rank < beam_size:
                        finished.append((tokens[1:], score / ((len(tokens) - 1) ** length_penalty)))
                elif len(next_active) < beam_size:
                    next_active.append((tokens, score))
            finished.sort(key=lambda item: item[1], reverse=True)
            finished = finished[:beam_size]
            if len(finished) >= beam_size or not next_active:
                break
            active = next_active
        if not finished:
            raise RuntimeError("decoder could not produce EOS within its configured limit")
        outputs.append(finished[0][0])
    width = max(map(len, outputs))
    result = src.new_full((len(outputs), width), PAD_ID)
    for row, values in enumerate(outputs):
        result[row, :len(values)] = src.new_tensor(values)
    return result
