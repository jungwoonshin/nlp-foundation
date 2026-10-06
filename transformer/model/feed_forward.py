"""Position-wise feed-forward signature, equation 2 in section 3.3."""
from torch import Tensor, nn


class PositionwiseFeedForward(nn.Module):
    """FFN(x)=ReLU(x W_1+b_1)W_2+b_2; (B,L,d_model) -> same shape."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.):
        super().__init__()
        raise NotImplementedError("Implement position-wise feed-forward layers")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
