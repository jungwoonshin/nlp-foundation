"""Encoder signatures: self-attention then position-wise FFN, each with post-LN."""
from torch import Tensor, nn
from transformer.config import TransformerConfig

from transformer.model.attention import MultiHeadAttention
from transformer.model.feed_forward import PositionwiseFeedForward
from transformer.model.normalization import LayerNorm

class EncoderLayer(nn.Module):
    """Self-attention then FFN, each post-LN; preserve (B,S,d_model)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.first_sublayer = MultiHeadAttention(config.d_model, heads=config.heads, dropout=config.attention_dropout)
        self.second_sublayer = PositionwiseFeedForward(config.d_model, d_ff=config.d_ff, dropout=config.activation_dropout)
        self.first_layer_norm = LayerNorm(config.d_model)
        self.second_layer_norm = LayerNorm(config.d_model)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x: Tensor, blocked_mask: Tensor) -> Tensor:
        """x=(B,S,d_model); blocked_mask=(B,1,1,S); output matches x."""
        attention_output = self.first_sublayer(x,x,x,blocked_mask).context
        x = self.first_layer_norm(x + self.dropout(attention_output))
        x = self.second_layer_norm(x + self.dropout(self.second_sublayer(x)))
        return x

class Encoder(nn.Module):
    """Section 3.1 encoder stack; no parameter sharing between layers."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ENCODER_LAYERS_NUM: int = config.encoder_layers
        self.encoder_modules = nn.ModuleList([EncoderLayer(config) for i in range(self.ENCODER_LAYERS_NUM)])

    def forward(self, x: Tensor, blocked_mask: Tensor) -> Tensor:
        """Stack encoder_layers independent layers, returning (B,S,d_model)."""
        for i in range(self.ENCODER_LAYERS_NUM):
            x = self.encoder_modules[i](x, blocked_mask)
        return x
