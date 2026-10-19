"""Decoder signatures: masked self-attention, cross-attention, FFN, each post-LN."""
from torch import Tensor, nn
from transformer.config import TransformerConfig

from transformer.model.attention import MultiHeadAttention
from transformer.model.feed_forward import PositionwiseFeedForward
from transformer.model.normalization import LayerNorm

class DecoderLayer(nn.Module):
    """Section 3.1's three post-LN sublayers; preserve (B,T,d_model)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.first_sublayer = MultiHeadAttention(config.d_model, heads=config.heads, dropout=config.attention_dropout)
        self.second_sublayer = MultiHeadAttention(config.d_model, heads=config.heads, dropout=config.attention_dropout)
        self.third_sublayer = PositionwiseFeedForward(config.d_model, d_ff=config.d_ff, dropout=config.activation_dropout)

        self.first_layernorm = LayerNorm(config.d_model)
        self.second_layernorm = LayerNorm(config.d_model)
        self.third_layernorm = LayerNorm(config.d_model)

        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x: Tensor, memory: Tensor, self_mask: Tensor, source_mask: Tensor) -> Tensor:
        """x=(B,T,d_model), memory=(B,S,d_model); return shape of x.

        self_mask=(B,1,T,T); source_mask=(B,1,1,S), both True for blocked keys.
        B=batch size, T=target length, S=source length, d_model=model width.
        Cross-attention Q comes from decoder, K/V from encoder memory.
        """
        self_attention_output = self.first_sublayer(x, x, x, self_mask).context  # (B,T,d_model)
        query = self.first_layernorm(x + self.dropout(self_attention_output))  # (B,T,d_model)
        
        cross_attention_output = self.second_sublayer(query, memory, memory, source_mask).context  # (B,T,d_model)
        x = self.second_layernorm(query + self.dropout(cross_attention_output))  # (B,T,d_model)
        
        ffn_x = self.third_sublayer(x)  # (B,T,d_model); internal FFN width is d_ff
        x = self.third_layernorm(x + self.dropout(ffn_x))  # (B,T,d_model)
        return x  # (B,T,d_model)


class Decoder(nn.Module):
    """Independent decoder layers with causal and encoder key-padding masks."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.DECODER_LAYERS_NUM: int = config.decoder_layers
        self.decoder_modules = nn.ModuleList([DecoderLayer(config) for i in range(self.DECODER_LAYERS_NUM)])

    def forward(self, x: Tensor, memory: Tensor, self_mask: Tensor, source_mask: Tensor) -> Tensor:
        """Stack decoder_layers independent layers, returning (B,T,d_model)."""
        for i in range(self.DECODER_LAYERS_NUM):
            x = self.decoder_modules[i](x, memory, self_mask, source_mask)
        return x
