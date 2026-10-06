"""Teacher-forced Transformer interface; loss and autoregressive search live outside."""
from torch import Tensor, nn
from transformer.config import TransformerConfig


class TransformerNMT(nn.Module):
    """Paper encoder/decoder with joined source/target/output embeddings.

    This scaffold intentionally cannot train a real Transformer yet. Implement
    components in model/ without nn.Transformer/MultiheadAttention shortcuts.
    Source EOS is included, src is left padded, and tgt_in starts with EOS
    (Fairseq decoder-start convention). There is no RNN state, source reversal,
    or Luong input feeding. PAD ids must be excluded by attention masks.
    """
    def __init__(self, vocab_size: int, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError("Transformer model math is a scaffold; implement model/ before real training")

    def encode(self, src: Tensor, src_lengths: Tensor) -> Tensor:
        """src=(B,S), lengths=(B,) including EOS -> memory=(B,S,d_model)."""
        raise NotImplementedError

    def forward(self, src: Tensor, src_lengths: Tensor, tgt_in: Tensor) -> Tensor:
        """Teacher forcing with causal mask; return logits (B,T,vocab_size)."""
        raise NotImplementedError
