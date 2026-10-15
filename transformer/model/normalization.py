"""Post-normalization signatures: section 3.1, LayerNorm(x+Dropout(Sublayer(x)))."""
from collections.abc import Callable
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """Normalize over d_model, using population variance and learned scale/shift."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x: Tensor) -> Tensor:
        # In forward; x: (B, L, d_model)
        output = self.norm(x)  # same shape
        return output

class ResidualConnection(nn.Module):
    """Post-LN residual connection; (B,L,d_model) -> same shape."""
    def __init__(self, d_model: int, dropout: float = .3):
        super().__init__()


        raise NotImplementedError("Implement post-normalization residual connections")

    def forward(self, x: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        raise NotImplementedError("Implement post-normalization residual connections")
