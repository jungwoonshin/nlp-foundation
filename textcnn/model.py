# Implement TextCNN here. train_textcnn.py and tests/test_textcnn.py expect this interface.
#
# class TextCNN(nn.Module):
#     def __init__(
#         self,
#         vocab_size: int,          # pad 포함: V + 1
#         embedding_dim: int,
#         num_classes: int,
#         filter_sizes: tuple[int, ...] = (3, 4, 5),
#         num_filters: int = 100,
#         dropout: float = 0.5,
#         padding_idx: int = 0,
#     ) -> None: ...
#
#     def forward(self, tokens: torch.Tensor) -> torch.Tensor:
#         # tokens: (batch, seq_len) long, pad = padding_idx
#         # return: logits (batch, num_classes)
#         ...
