"""Left-padded sources, right-padded targets, and length-grouped token-budget batches."""
from dataclasses import dataclass
import torch
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


class TokenBatchSampler(Sampler):
    def __init__(self, dataset, max_tokens=4096, *, shuffle=True, seed=1, batch_size=None):
        self.dataset, self.max_tokens = dataset, max_tokens
        self.shuffle, self.seed, self.batch_size = shuffle, seed, batch_size
        if max_tokens < 1 or (batch_size is not None and batch_size < 1):
            raise ValueError("batch limits must be positive")

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        indices = (torch.randperm(len(self.dataset), generator=generator).tolist()
                   if self.shuffle else list(range(len(self.dataset))))
        if self.shuffle:
            indices.sort(key=lambda i: max(len(self.dataset.rows[i][0]), len(self.dataset.rows[i][1]) + 1))
        batches, current, width = [], [], 0
        for i in indices:
            src, tgt = self.dataset.rows[i]
            size = max(len(src), len(tgt) + 1)
            if size > self.max_tokens:
                raise ValueError("one example exceeds token budget")
            if current and ((max(width, size) * (len(current) + 1) > self.max_tokens)
                            or (self.batch_size is not None and len(current) >= self.batch_size)):
                batches.append(current)
                current, width = [], 0
            current.append(i)
            width = max(width, size)
        if current:
            batches.append(current)
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]
        yield from batches

    def __len__(self):
        return sum(1 for _ in self)


@dataclass
class ProcessedParallel:
    dataset: ParallelDataset
    vocab: Vocab

    def dataloader(self, max_tokens=4096, *, shuffle=False, seed=1, batch_size=None):
        return DataLoader(self.dataset, batch_sampler=TokenBatchSampler(
            self.dataset, max_tokens, shuffle=shuffle, seed=seed, batch_size=batch_size),
            collate_fn=pad_collate, generator=torch.Generator().manual_seed(seed))


def process_pairs(pairs, vocab, max_positions=1024):
    return ProcessedParallel(ParallelDataset(pairs, vocab, max_positions), vocab)
