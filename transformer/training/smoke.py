"""Small trainable infrastructure fixture, deliberately NOT a Transformer."""
from torch import nn


class DummyTransformer(nn.Module):
    """Embeddings and a linear output head; no attention, positions, or encoder stack."""
    def __init__(self, vocab_size, d_model=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_lengths, tgt_in):
        return self.output(self.embedding(tgt_in))
