"""Teacher-forced Transformer interface; loss and autoregressive search live outside."""
from torch import Tensor, nn
from transformer.config import TransformerConfig

from transformer.data.vocab import PAD_ID
from transformer.data.masks import decoder_mask

from transformer.model.encoder import Encoder
from transformer.model.decoder import Decoder
from transformer.model.embeddings import TokenEmbedding
from transformer.model.positional_encoding import SinusoidalPositionalEncoding

class TransformerNMT(nn.Module):
    """Paper encoder/decoder with joined source/target/output embeddings.

    Components implement the pinned Multi30k text-only reference without
    nn.Transformer/MultiheadAttention shortcuts.
    Source EOS is included, src is left padded, and tgt_in starts with EOS
    (Fairseq decoder-start convention). There is no RNN state, source reversal,
    or Luong input feeding. PAD ids must be excluded by attention masks.
    """
    def __init__(self, vocab_size: int, config: TransformerConfig):
        super().__init__()
        self.token_embeddings = TokenEmbedding(vocab_size, d_model=config.d_model, padding_idx=PAD_ID) 
        self.positions = SinusoidalPositionalEncoding(
            config.d_model, config.max_positions, config.dropout
        )
        self.encoder: nn.Module = Encoder(config)
        self.decoder: nn.Module = Decoder(config)
        self.output_projection = nn.Linear(config.d_model, vocab_size, bias=False)
        self.output_projection.weight = self.token_embeddings.embedding.weight # Tie weights between Embedding and Output Projection

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """src=(B,S), src_mask=(B,1,1,S) -> memory=(B,S,d_model)."""
        src_embeddings = self.positions(self.token_embeddings(src), src.eq(PAD_ID))
        encoded = self.encoder(src_embeddings, src_mask)
        return encoded

    def decode(self, tgt_in: Tensor, memory: Tensor, source_mask: Tensor) -> Tensor:
        """tgt_in=(B,T), memory=(B,S,d_model),
            source_mask=(B,1,1,S), True at source PAD; 
            return decoder features (B,T,d_model)."""
        # x: Tensor, memory: Tensor, self_mask: Tensor, source_mask: Tensor
        tgt_in_embeddings = self.positions(self.token_embeddings(tgt_in), tgt_in.eq(PAD_ID))
        self_mask = decoder_mask(tgt_in)
        decoded = self.decoder(tgt_in_embeddings, memory, self_mask, source_mask)
        return decoded

    def forward(self, src: Tensor, src_lengths: Tensor, tgt_in: Tensor) -> Tensor:
        """Teacher forcing with causal mask; return logits (B,T,vocab_size)."""
        src_mask = (src == PAD_ID)[:, None, None, :]
        encoded = self.encode(src, src_mask)
        decoded = self.decode(tgt_in, encoded, src_mask)
        out = self.output_projection(decoded)
        return out
