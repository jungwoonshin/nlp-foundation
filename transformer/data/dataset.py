"""Left-padded sources, right-padded targets, and length-grouped token-budget batches."""
from dataclasses import dataclass
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from transformer.data.vocab import EOS_ID, PAD_ID, Vocab


class ParallelDataset(Dataset):
    def __init__(self, pairs, vocab: Vocab, max_positions=1024):
        self.pairs, self.vocab = pairs, vocab
        if not pairs:
            raise ValueError("empty dataset")
        self.rows = []
        for src, tgt in pairs:
            if not src or not tgt:
                raise ValueError("empty content row")
            if max(len(src), len(tgt)) + 1 > max_positions:
                raise ValueError("sequence exceeds max_positions; no silent truncation")
            self.rows.append((vocab.encode(src) + [EOS_ID], vocab.encode(tgt)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        src, tgt = self.rows[index]
        return {"src": torch.tensor(src), "tgt_in": torch.tensor([EOS_ID, *tgt]),
                "tgt_out": torch.tensor([*tgt, EOS_ID])}


def pad_collate(rows):
    if not rows:
        raise ValueError("empty batch")
    b, s, t = len(rows), max(len(r["src"]) for r in rows), max(len(r["tgt_in"]) for r in rows)
    batch = {"src": torch.full((b, s), PAD_ID, dtype=torch.long),
             "tgt_in": torch.full((b, t), PAD_ID, dtype=torch.long),
             "tgt_out": torch.full((b, t), PAD_ID, dtype=torch.long),
             "src_lengths": torch.tensor([len(r["src"]) for r in rows])}
    for i, row in enumerate(rows):
        batch["src"][i, -len(row["src"]):] = row["src"]
        for name in ("tgt_in", "tgt_out"):
            batch[name][i, :len(row[name])] = row[name]
    return batch


def reference_collate(rows):
    """Fairseq sorts each mini-batch by descending source length."""
    return pad_collate(sorted(rows, key=lambda row: len(row['src']), reverse=True))


class TokenBatchSampler(Sampler):
    """Pinned Fairseq stable length ordering, batch multiples, and epoch shuffling."""
    def __init__(self, dataset, max_tokens=4096, *, shuffle=True, seed=1,
                 batch_size=None, epoch=1, required_multiple=8):
        self.dataset, self.max_tokens = dataset, max_tokens
        self.shuffle, self.seed, self.batch_size = shuffle, seed, batch_size
        self.epoch, self.required_multiple = epoch, required_multiple
        if max_tokens < 1 or (batch_size is not None and batch_size < 1):
            raise ValueError("batch limits must be positive")

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(len(self.dataset)) if self.shuffle else np.arange(len(self.dataset))
        src = np.array([len(r[0]) for r in self.dataset.rows])
        tgt = np.array([len(r[1]) + 1 for r in self.dataset.rows])
        indices = indices[np.argsort(tgt[indices], kind='mergesort')]
        indices = indices[np.argsort(src[indices], kind='mergesort')]
        batches, current, sizes = [], [], []
        for index in indices:
            i = int(index)
            size = int(max(src[i], tgt[i]))
            if size > self.max_tokens:
                raise ValueError("one example exceeds token budget")
            sizes.append(size)
            if current and (max(sizes) * (len(current) + 1) > self.max_tokens
                            or (self.batch_size is not None and len(current) >= self.batch_size)):
                m = self.required_multiple
                split = max(m * (len(current) // m), len(current) % m)
                batches.append(current[:split])
                current, sizes = current[split:], sizes[split:]
            current.append(i)
        if current:
            batches.append(current)
        if self.shuffle:
            order = np.random.RandomState(self.seed + self.epoch).permutation(len(batches))
            batches = [batches[i] for i in order]
        yield from batches

    def __len__(self):
        return sum(1 for _ in self)


@dataclass
class ProcessedParallel:
    dataset: ParallelDataset
    vocab: Vocab

    def dataloader(self, max_tokens=4096, *, shuffle=False, seed=1, batch_size=None, epoch=1):
        return DataLoader(self.dataset, batch_sampler=TokenBatchSampler(
            self.dataset, max_tokens, shuffle=shuffle, seed=seed, batch_size=batch_size, epoch=epoch),
            collate_fn=reference_collate, generator=torch.Generator().manual_seed(seed))


def process_pairs(pairs, vocab, max_positions=1024):
    return ProcessedParallel(ParallelDataset(pairs, vocab, max_positions), vocab)
