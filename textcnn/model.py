# Implement TextCNN here. scripts/textcnn/train and tests/test_textcnn.py expect this interface.
#
import torch
from torch import nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,          # pad 포함: V + 1
        embedding_dim: int,
        num_classes: int,
        filter_sizes: tuple[int, ...] = (3, 4, 5),
        num_filters: int = 100,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        # The purpose of padding_idx is to set the embedding of padding tokens to 0.
        # The effect is that the padding tokens are not considered when calculating the loss.
        # This is useful because we want to ignore the padding tokens when calculating the loss.
        # If we don't set the padding tokens to 0, the loss will be calculated based on the padding tokens.
        # This is not what we want because we want to ignore the padding tokens when calculating the loss.
        # So we set the padding tokens to 0.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # Kim CNN-rand: U[-0.25, 0.25] so random vectors match word2vec-scale variance.
        nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        with torch.no_grad():
            self.embedding.weight[padding_idx].zero_()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (size, embedding_dim)) for size in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (batch, seq_len) long, pad = padding_idx
        # return: logits (batch, num_classes)
        # Dropout is only on the penultimate layer (Kim 2014 §2.1), not embeddings.
        embedded = self.embedding(tokens)  # (batch, seq_len, embedding_dim)
        embedded = embedded.unsqueeze(1)  # (batch, 1, seq_len, embedding_dim)

        # After conv2d operation, we have a 4D tensor (batch, num_filters, seq_len - filter_size + 1, 1).
        # Last embedding dimension is 1 because filter's width is embedding_dim and filter's height is filter_size.
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs] # (batch, num_filters, seq_len - filter_size + 1)
        
        # max_pool1d takes input (batch, channels, length) and output (batch, channels, 1) length is gone because max pooling is applied to the length dimension.
        # So we squeeze(2) to remove the length dimension.
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] # [(batch, num_filters), (batch, num_filters), (batch, num_filters)]
        cat = self.dropout(torch.cat(pooled, dim=1)) # (batch, num_filters * len(filter_sizes))
        return self.fc(cat) # (batch, num_classes)