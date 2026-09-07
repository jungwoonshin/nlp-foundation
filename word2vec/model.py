"""Minimal MLP classifier on tutorial/sample_data.csv (XOR-style 2D points)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = Path(__file__).resolve().parent / "sample_data.csv"

class Word2Vec(nn.Module):
    
    # dot-product between the center and context word embeddings
    # either use hierchical softmax or negative sampling to train the model
    # word embeddings are initialized to random vectors with 2 embedding lookup banks initialzation
    # the model is trained to predict the context word given the center word

    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 100) -> None:
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center_embeddings = self.center_embeddings(x)
        context_embeddings = self.context_embeddings(x)
        return torch.bmm(center_embeddings, context_embeddings.transpose(1, 2))
