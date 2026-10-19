"""Position-wise feed-forward signature, equation 2 in section 3.3."""
from torch import Tensor, nn


class PositionwiseFeedForward(nn.Module):
    """FFN(x)=ReLU(x W_1+b_1)W_2+b_2; (B,L,d_model) -> same shape."""
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.):
        super().__init__()
        self.D_FF = d_ff
        self.D_MODEL = d_model
        self.projection_one = nn.Linear(self.D_MODEL, self.D_FF)
        nn.init.xavier_uniform_(self.projection_one.weight)
        nn.init.zeros_(self.projection_one.bias)
        self.projection_two = nn.Linear(self.D_FF, self.D_MODEL)
        nn.init.xavier_uniform_(self.projection_two.weight)
        nn.init.zeros_(self.projection_two.bias)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        projected = self.projection_one(x)
        projected = self.dropout(self.activation(projected))
        projected = self.projection_two(projected)
        return projected
