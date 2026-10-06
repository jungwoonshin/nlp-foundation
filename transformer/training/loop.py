"""Token-normalized accumulation, reference Adam schedule, and resumable epochs."""
from dataclasses import asdict
import itertools
from pathlib import Path
import torch
from transformer.config import TransformerConfig
from transformer.training.checkpoints import load_checkpoint, save_checkpoint
from transformer.training.loss import token_loss
from transformer.training.schedule import learning_rate
from transformer.eval.metrics import evaluate_perplexity


def train_update(model, batches, optimizer, config, device):
    """One optimizer update from summed token losses across unequal microbatches."""
    optimizer.zero_grad(set_to_none=True)
    total_loss = total_nll = total_tokens = 0
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["src"], batch["src_lengths"], batch["tgt_in"])
        loss, nll, count = token_loss(logits, batch["tgt_out"], config.label_smoothing)
        loss.backward()
        total_loss += float(loss.detach())
        total_nll += float(nll.detach())
        total_tokens += count
    if not total_tokens:
        raise ValueError("optimizer update has no valid target tokens")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(total_tokens)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": total_loss / total_tokens, "nll": total_nll / total_tokens,
            "tokens": total_tokens, "grad_norm": float(norm)}


def fit(model, train, valid, config: TransformerConfig, device, *, output_dir,
        metadata=None, resume=None, verbose=True):
    config.validate()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(0, config),
                                 betas=config.adam_betas, eps=config.adam_eps)
    metadata = dict(metadata or {})
    metadata.setdefault("config", asdict(config))
    metadata.setdefault("vocab", train.vocab.to_dict())
    state = {"epoch": 1, "next_batch": 0, "updates": 0,
             "best_loss": float("inf"), "bad_epochs": 0, "history": []}
    if resume is not None:
        saved = load_checkpoint(resume)
        old_config = dict(saved["metadata"]["config"])
        new_config = asdict(config)
        for limits in ("max_updates", "max_epochs"):
            old_config.pop(limits, None)
            new_config.pop(limits, None)
        if old_config != new_config or saved["metadata"]["vocab"] != train.vocab.to_dict():
            raise ValueError("resume configuration/vocabulary mismatch")
        for key in ("data_manifest", "code_sha256"):
            if saved["metadata"].get(key) != metadata.get(key):
                raise ValueError(f"resume {key} mismatch")
        load_checkpoint(resume, model=model, optimizer=optimizer, restore_rng=True)
        state = saved["state"]
        metadata = {**saved["metadata"], "config": asdict(config)}
    output_dir = Path(output_dir)
    while (state["epoch"] <= config.max_epochs and state["updates"] < config.max_updates
           and state["bad_epochs"] < config.patience):
        epoch = state["epoch"]
        model.train()
        loader = train.dataloader(config.max_tokens, shuffle=True, seed=config.seed + epoch,
                                  batch_size=config.batch_size)
        total_batches = len(loader)
        iterator = itertools.islice(iter(loader), state["next_batch"], None)
        while group := list(itertools.islice(iterator, config.accumulation_steps)):
            for parameters in optimizer.param_groups:
                parameters["lr"] = learning_rate(state["updates"], config)
            train_update(model, group, optimizer, config, device)
            state["updates"] += 1
            state["next_batch"] += len(group)
            if state["updates"] >= config.max_updates:
                break
        metrics = evaluate_perplexity(model, valid, device, max_tokens=config.max_tokens,
                                      batch_size=config.batch_size, epsilon=config.label_smoothing)
        completed = state["next_batch"] >= total_batches
        if completed:
            state["bad_epochs"] = 0 if metrics["mean_loss"] < state["best_loss"] else state["bad_epochs"] + 1
            state["best_loss"] = min(state["best_loss"], metrics["mean_loss"])
            state["epoch"], state["next_batch"] = epoch + 1, 0
        state["history"].append({"epoch": epoch, "updates": state["updates"],
                                 "epoch_complete": completed, "validation": metrics})
        filename = f"checkpoint_epoch_{epoch:04d}.pt" if completed else "checkpoint_partial.pt"
        save_checkpoint(output_dir / filename, model, optimizer, state, metadata)
        if verbose:
            print(f"epoch={epoch} updates={state['updates']} valid_ppl={metrics['perplexity']:.4f} complete={completed}")
    return state
