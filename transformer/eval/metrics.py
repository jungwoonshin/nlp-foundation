"""Teacher-forced unsmoothed PPL and separately decoded corpus BLEU."""
import math
import torch
from transformer.training.loss import token_loss
from transformer.eval.bleu import corpus_bleu, remove_bpe
from transformer.eval.decoding import generate


@torch.no_grad()
def evaluate_perplexity(model, processed, device, *, max_tokens=4096, batch_size=None, epsilon=0.):
    model.eval()
    nll = tokens = smoothed_loss = 0
    for batch in processed.dataloader(max_tokens, batch_size=batch_size):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["src"], batch["src_lengths"], batch["tgt_in"])
        smoothed, loss, count = token_loss(logits, batch["tgt_out"], epsilon)
        smoothed_loss += float(smoothed)
        nll += float(loss)
        tokens += count
    if not tokens:
        raise ValueError("no valid target tokens for perplexity")
    mean = nll / tokens
    return {"perplexity": math.exp(mean) if mean < 709 else float("inf"),
            "nll": nll, "tokens": tokens, "mean_nll": mean, "mean_loss": smoothed_loss / tokens}


@torch.no_grad()
def evaluate_bleu(model, processed, device, *, beam_size=5, max_decode_tokens=200,
                  length_penalty=1., max_tokens=4096):
    hypotheses, references = [], []
    for batch in processed.dataloader(max_tokens, batch_size=128):
        prediction = generate(model, batch["src"].to(device), batch["src_lengths"].to(device),
                              beam_size=beam_size, max_tokens=max_decode_tokens,
                              length_penalty=length_penalty)
        hypotheses.extend(remove_bpe(processed.vocab.decode(row)) for row in prediction)
        references.extend(remove_bpe(processed.vocab.decode(row, reference=True)) for row in batch["tgt_out"])
    return {"bleu": corpus_bleu(hypotheses, references), "hypotheses": hypotheses, "references": references}
