# Complete Transformer code changes: before and after

Before: Git HEAD `4ace440fb801d7ac9c989a353d1505bbff59e1a2`. After: working-tree implementation frozen for the 2026-10-19 Multi30k run.

Every section includes the complete file before and after the change, including unchanged lines. No code excerpts or ellipses are used. Files added in this task have no prior version. Full files are used so multiple changes to the same function remain in context.

The four feature areas are model, data, training and evaluation. Validation changes are included separately. Documentation-only additions such as REFERENCE.md and this report are not executable code and are listed at the end.

Color key: in the rendered Markdown preview, `-` lines are removed (red), `+` lines are added (green), and lines beginning with a space are unchanged. Edited lines appear as red in Before and green in After. These are diff markers, not part of the Python code. Color requires a Markdown viewer with diff syntax highlighting; the markers remain visible even when colors are unavailable.

## Model: token embedding initialization

File: `transformer/model/embeddings.py`

Token embeddings now start from a normal distribution whose standard deviation is the inverse square root of the model width. The padding row is explicitly initialized to zero so padding tokens begin with no embedding signal. These changes match the reference initialization while preserving lookup behavior and embedding scaling.

### Before

```diff
 """Embedding signatures: paper section 3.4; reference shares all three weight matrices."""
 from torch import Tensor, nn
 import math
 
 class TokenEmbedding(nn.Module):
     """tokens (B,L) -> sqrt(d_model) * embeddings (B,L,d_model); PAD row is zero."""
     def __init__(self, vocab_size: int, d_model: int = 512, padding_idx: int = 1):
         super().__init__()
         self.D_MODEL = d_model
         self.embedding = nn.Embedding(vocab_size, self.D_MODEL, padding_idx=padding_idx)
         self.NORMALIZE_CONSTANT = math.sqrt(self.D_MODEL)
 
     def forward(self, tokens: Tensor) -> Tensor:
         return self.embedding(tokens) * self.NORMALIZE_CONSTANT
```

### After

```diff
 """Embedding signatures: paper section 3.4; reference shares all three weight matrices."""
 from torch import Tensor, nn
+import torch
 import math
 
 class TokenEmbedding(nn.Module):
     """tokens (B,L) -> sqrt(d_model) * embeddings (B,L,d_model); PAD row is zero."""
     def __init__(self, vocab_size: int, d_model: int = 512, padding_idx: int = 1):
         super().__init__()
         self.D_MODEL = d_model
         self.embedding = nn.Embedding(vocab_size, self.D_MODEL, padding_idx=padding_idx)
+        nn.init.normal_(self.embedding.weight, mean=0., std=d_model ** -0.5)
+        with torch.no_grad():
+            self.embedding.weight[padding_idx].zero_()
         self.NORMALIZE_CONSTANT = math.sqrt(self.D_MODEL)
 
     def forward(self, tokens: Tensor) -> Tensor:
         return self.embedding(tokens) * self.NORMALIZE_CONSTANT
```

## Model: attention biases and initialization

File: `transformer/model/attention.py`

The query, key, value and output projections now include the biases used by the reference implementation. Query, key and value weights use scaled Xavier initialization, while the output projection uses Xavier initialization with a zero bias. Construction order also follows the reference so the same random seed can reproduce the tested initial projection parameters.

### Before

```diff
 """Scaled dot-product and multi-head attention for paper equations 1 and 2."""
 from torch import Tensor, nn
 from transformer.model.types import AttentionOutput
 
 import math, torch
 
 def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor,
                                  blocked_mask: Tensor | None = None,
                                  dropout_p: float = 0., training: bool = True) -> AttentionOutput:
     """softmax(QK^T/sqrt(d_k))V, masking before softmax.
 
     q=(B,h,Q,d_k), k=(B,h,K,d_k), v=(B,h,K,d_v).
     blocked_mask broadcasts to (B,h,Q,K); True means blocked. Return context
     (B,h,Q,d_v) and weights (B,h,Q,K). All-blocked queries must not produce NaN.
     Returned weights include attention dropout, applied only during training.
     """
 
     sqrt_d_k = math.sqrt(k.shape[-1])
     scores = q @ k.transpose(-2, -1) / sqrt_d_k # (B, h, Q, K)
 
     if blocked_mask is not None:
         scores = scores.masked_fill(blocked_mask, float("-inf"))
         all_blocked = blocked_mask.all(dim=-1, keepdim=True)
         scores = scores.masked_fill(all_blocked, 0.0)
 
     weights = torch.softmax(scores, dim=-1)# (B, h, Q, K)
 
     if blocked_mask is not None:
         weights = weights.masked_fill(blocked_mask, 0.0)
 
     weights = torch.nn.functional.dropout(weights, p=dropout_p, training=training)
     context = weights @ v # (B, h, Q, d_v)
     return AttentionOutput(context=context, weights=weights)
 
 
 class MultiHeadAttention(nn.Module):
     """Learn Q/K/V projections, concatenate heads, and project through W_O.
 
     Query/key/value: (B,L,d_model). Return context (B,Q,d_model), per-head
     weights (B,h,Q,K). Match reference linear biases and initialization;
     attention dropout is zero in the Multi30k reference profile.
     """
     def __init__(self, d_model: int, heads: int, dropout: float = 0.):
         super().__init__()
 
         if d_model <= 0:
             raise ValueError("d_model must be positive")
         if heads <= 0:
             raise ValueError("heads must be positive")
         if d_model % heads != 0:
             raise ValueError("d_model must be divisible by heads")
         if not 0 <= dropout <= 1:
             raise ValueError("dropout must be between zero and one")
 
         self.dropout = dropout
         self.heads = heads
         self.d_k = d_model // heads
         self.d_v = d_model // heads
 
-        self.q_proj = torch.nn.Linear(d_model, heads * self.d_k, bias=False)
-        self.k_proj = torch.nn.Linear(d_model, heads * self.d_k, bias=False)
-        self.v_proj = torch.nn.Linear(d_model, heads * self.d_v, bias=False)
-        self.o_proj = torch.nn.Linear(heads * self.d_v, d_model, bias=False)
 
     def forward(self, query: Tensor, key: Tensor, value: Tensor,
                 blocked_mask: Tensor | None = None) -> AttentionOutput:
 
         query, key, value = self.q_proj(query), self.k_proj(key), self.v_proj(value)
         # query/key is (B,Q/K,heads * self.d_k), and value is (B,K,heads * self.d_v)
         B, Q, _ = query.shape
         K = key.size(1)
         h = self.heads
 
         q = query.reshape(B, Q, h, self.d_k).transpose(1, 2)
         k = key.reshape(B, K, h, self.d_k).transpose(1, 2)
         v = value.reshape(B, K, h, self.d_v).transpose(1, 2)
 
         attention_output = scaled_dot_product_attention(
             q, k, v, blocked_mask, dropout_p=self.dropout, training=self.training
         )
         context = attention_output.context.transpose(1, 2)
         context = context.reshape(B, Q, h * self.d_v)
         context = self.o_proj(context)
 
         return AttentionOutput(
             context=context,                   # (B, Q, d_model)
             weights=attention_output.weights,  # (B, h, Q, K)
         )
         
      
```

### After

```diff
 """Scaled dot-product and multi-head attention for paper equations 1 and 2."""
 from torch import Tensor, nn
 from transformer.model.types import AttentionOutput
 
 import math, torch
 
 def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor,
                                  blocked_mask: Tensor | None = None,
                                  dropout_p: float = 0., training: bool = True) -> AttentionOutput:
     """softmax(QK^T/sqrt(d_k))V, masking before softmax.
 
     q=(B,h,Q,d_k), k=(B,h,K,d_k), v=(B,h,K,d_v).
     blocked_mask broadcasts to (B,h,Q,K); True means blocked. Return context
     (B,h,Q,d_v) and weights (B,h,Q,K). All-blocked queries must not produce NaN.
     Returned weights include attention dropout, applied only during training.
     """
 
     sqrt_d_k = math.sqrt(k.shape[-1])
     scores = q @ k.transpose(-2, -1) / sqrt_d_k # (B, h, Q, K)
 
     if blocked_mask is not None:
         scores = scores.masked_fill(blocked_mask, float("-inf"))
         all_blocked = blocked_mask.all(dim=-1, keepdim=True)
         scores = scores.masked_fill(all_blocked, 0.0)
 
     weights = torch.softmax(scores, dim=-1)# (B, h, Q, K)
 
     if blocked_mask is not None:
         weights = weights.masked_fill(blocked_mask, 0.0)
 
     weights = torch.nn.functional.dropout(weights, p=dropout_p, training=training)
     context = weights @ v # (B, h, Q, d_v)
     return AttentionOutput(context=context, weights=weights)
 
 
 class MultiHeadAttention(nn.Module):
     """Learn Q/K/V projections, concatenate heads, and project through W_O.
 
     Query/key/value: (B,L,d_model). Return context (B,Q,d_model), per-head
     weights (B,h,Q,K). Match reference linear biases and initialization;
     attention dropout is zero in the Multi30k reference profile.
     """
     def __init__(self, d_model: int, heads: int, dropout: float = 0.):
         super().__init__()
 
         if d_model <= 0:
             raise ValueError("d_model must be positive")
         if heads <= 0:
             raise ValueError("heads must be positive")
         if d_model % heads != 0:
             raise ValueError("d_model must be divisible by heads")
         if not 0 <= dropout <= 1:
             raise ValueError("dropout must be between zero and one")
 
         self.dropout = dropout
         self.heads = heads
         self.d_k = d_model // heads
         self.d_v = d_model // heads
 
+        # Match the reference's construction and initialization order.
+        self.k_proj = nn.Linear(d_model, heads * self.d_k)
+        self.v_proj = nn.Linear(d_model, heads * self.d_v)
+        self.q_proj = nn.Linear(d_model, heads * self.d_k)
+        self.o_proj = nn.Linear(heads * self.d_v, d_model)
+        for layer in (self.k_proj, self.v_proj, self.q_proj):
+            nn.init.xavier_uniform_(layer.weight, gain=1 / math.sqrt(2))
+        nn.init.xavier_uniform_(self.o_proj.weight)
+        nn.init.zeros_(self.o_proj.bias)
 
     def forward(self, query: Tensor, key: Tensor, value: Tensor,
                 blocked_mask: Tensor | None = None) -> AttentionOutput:
 
         query, key, value = self.q_proj(query), self.k_proj(key), self.v_proj(value)
         # query/key is (B,Q/K,heads * self.d_k), and value is (B,K,heads * self.d_v)
         B, Q, _ = query.shape
         K = key.size(1)
         h = self.heads
 
         q = query.reshape(B, Q, h, self.d_k).transpose(1, 2)
         k = key.reshape(B, K, h, self.d_k).transpose(1, 2)
         v = value.reshape(B, K, h, self.d_v).transpose(1, 2)
 
         attention_output = scaled_dot_product_attention(
             q, k, v, blocked_mask, dropout_p=self.dropout, training=self.training
         )
         context = attention_output.context.transpose(1, 2)
         context = context.reshape(B, Q, h * self.d_v)
         context = self.o_proj(context)
 
         return AttentionOutput(
             context=context,                   # (B, Q, d_model)
             weights=attention_output.weights,  # (B, h, Q, K)
         )
         
      
```

## Model: Fairseq positional encoding

File: `transformer/model/positional_encoding.py`

The positional table now groups sine channels before cosine channels and uses the reference frequency spacing, with a final zero channel for odd model widths. Cumulative counts of non-padding tokens assign positions independently of left or right padding, reserving a zero table row at the padding index. The selected positional vectors are cast to the embedding dtype, added to the embeddings and passed through dropout.

### Before

```diff
-"""Sinusoidal position signatures: section 3.5, with padding-aware reference coordinates."""
 from torch import Tensor, nn
 
-import torch
 
 class SinusoidalPositionalEncoding(nn.Module):
-    """Add sinusoidal positions to (B,L,d_model).
 
-    Paper: PE(pos,2i)=sin(pos/10000**(2i/d_model)), odd channels use cosine.
-    The small-data reference uses Fairseq sinusoidal channel layout/frequencies,
-    non-PAD cumulative positions plus padding_idx, and zero vectors at PAD.
-    Implement the pinned reference convention for multi30k_tiny, documenting
-    its difference from the paper's interleaved layout. No learned positions.
     """
-
     positional_encoder: Tensor
 
-    def __init__(self, d_model: int, max_positions: int = 1024, dropout: float = .3):
         super().__init__()
-        # (pos, i)
-        pos = torch.arange(max_positions, dtype=torch.float32)[:, None]  # (POS, 1)
-        dims = torch.arange(0, d_model, 2, dtype=torch.float32)[None, :] # (1, D_MODEL//2)
-        # i = torch.arange(d_model, dtype=torch.float32)[None, :]      # (1, I)
-        angles = pos / (10000.0 ** (dims / d_model)) # (POS, I)
-
-        pe = torch.empty(max_positions, d_model, device=angles.device, dtype=angles.dtype)
-        pe[:, 0::2] = torch.sin(angles)  # dimensions 0, 2, 4, ...
-        pe[:, 1::2] = torch.cos(angles[:, :d_model // 2])  # dimensions 1, 3, 5, ...
         self.register_buffer("positional_encoder", pe)
         self.dropout = nn.Dropout(dropout)
 
-    # x: (B,L,d_model)
     def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
-        """padding_mask: (B,L), True at PAD; zero the full PAD vectors.
-
-        Apply dropout to embeddings plus positions; output has the shape of x.
-        """
-
-        # padding_mask: (B, L), True at PAD
         non_pad = (~padding_mask).long()
-        position_ids = (non_pad.cumsum(dim=1) - 1).clamp_min(0)
-
-        positions = self.positional_encoder[position_ids]
-        positions = positions.masked_fill(padding_mask.unsqueeze(-1), 0.0)
-
-        return self.dropout(x + positions)
```

### After

```diff
+"""Padding-aware sinusoidal positions matching the pinned Revisit-MMT/Fairseq code."""
+import math
+import torch
 from torch import Tensor, nn
+from transformer.data.vocab import PAD_ID
 
 
 class SinusoidalPositionalEncoding(nn.Module):
+    """Add concatenated sine/cosine channels, then embedding dropout.
 
+    Non-PAD positions begin at padding_idx + 1. PAD contributes no positions;
+    its input embedding is preserved. This differs from paper interleaving.
     """
     positional_encoder: Tensor
 
+    def __init__(self, d_model: int, max_positions: int = 1024,
+                 dropout: float = .3, padding_idx: int = PAD_ID):
         super().__init__()
+        if d_model < 4 or max_positions < 1:
+            raise ValueError("reference sinusoids require d_model >= 4 and max_positions > 0")
+        self.padding_idx, self.max_positions = padding_idx, max_positions
+        half = d_model // 2
+        frequencies = torch.exp(torch.arange(half, dtype=torch.float32)
+                                * (-math.log(10000.) / (half - 1)))
+        positions = torch.arange(max_positions + padding_idx + 1, dtype=torch.float32)
+        angles = positions[:, None] * frequencies[None, :]
+        pe = torch.cat((angles.sin(), angles.cos()), dim=1)
+        if d_model % 2:
+            pe = torch.cat((pe, torch.zeros(len(positions), 1)), dim=1)
+        pe[padding_idx].zero_()
         self.register_buffer("positional_encoder", pe)
         self.dropout = nn.Dropout(dropout)
 
     def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
         non_pad = (~padding_mask).long()
+        ids = non_pad.cumsum(dim=1) * non_pad + self.padding_idx
+        return self.dropout(x + self.positional_encoder[ids].to(x.dtype))
```

## Model: FFN initialization and activation dropout

File: `transformer/model/feed_forward.py`

Both feed-forward linear layers now use Xavier weights and zero biases in the reference construction order. Activation dropout is applied after ReLU, between the expansion and projection back to the model width. The Multi30k profile sets activation dropout to zero, while decoder and encoder sublayers apply their separate residual dropout.

### Before

```diff
 """Position-wise feed-forward signature, equation 2 in section 3.3."""
 from torch import Tensor, nn
 
 
 class PositionwiseFeedForward(nn.Module):
     """FFN(x)=ReLU(x W_1+b_1)W_2+b_2; (B,L,d_model) -> same shape."""
     def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.):
         super().__init__()
         self.D_FF = d_ff
         self.D_MODEL = d_model
         self.projection_one = nn.Linear(self.D_MODEL, self.D_FF)
         self.projection_two = nn.Linear(self.D_FF, self.D_MODEL)
         self.activation = nn.ReLU()
 
     def forward(self, x: Tensor) -> Tensor:
         projected = self.projection_one(x)
-        projected = self.activation(projected)
         projected = self.projection_two(projected)
         return projected
```

### After

```diff
 """Position-wise feed-forward signature, equation 2 in section 3.3."""
 from torch import Tensor, nn
 
 
 class PositionwiseFeedForward(nn.Module):
     """FFN(x)=ReLU(x W_1+b_1)W_2+b_2; (B,L,d_model) -> same shape."""
     def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.):
         super().__init__()
         self.D_FF = d_ff
         self.D_MODEL = d_model
         self.projection_one = nn.Linear(self.D_MODEL, self.D_FF)
+        nn.init.xavier_uniform_(self.projection_one.weight)
+        nn.init.zeros_(self.projection_one.bias)
         self.projection_two = nn.Linear(self.D_FF, self.D_MODEL)
+        nn.init.xavier_uniform_(self.projection_two.weight)
+        nn.init.zeros_(self.projection_two.bias)
         self.activation = nn.ReLU()
+        self.dropout = nn.Dropout(dropout)
 
     def forward(self, x: Tensor) -> Tensor:
         projected = self.projection_one(x)
+        projected = self.dropout(self.activation(projected))
         projected = self.projection_two(projected)
         return projected
```

## Model: LayerNorm epsilon argument

File: `transformer/model/normalization.py`

The LayerNorm wrapper now passes its epsilon argument to the underlying PyTorch layer instead of using a hardcoded value. Epsilon stabilizes normalization by being added to the variance before taking the square root. Default behavior is preserved, and callers can now select a different epsilon when matching another configuration.

### Before

```diff
 """Post-normalization signatures: section 3.1, LayerNorm(x+Dropout(Sublayer(x)))."""
 from collections.abc import Callable
 from torch import Tensor, nn
 
 
 class LayerNorm(nn.Module):
     """Normalize over d_model, using population variance and learned scale/shift."""
     def __init__(self, d_model: int, eps: float = 1e-5):
         super().__init__()
-        self.norm = nn.LayerNorm(d_model, eps=1e-5)
 
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
```

### After

```diff
 """Post-normalization signatures: section 3.1, LayerNorm(x+Dropout(Sublayer(x)))."""
 from collections.abc import Callable
 from torch import Tensor, nn
 
 
 class LayerNorm(nn.Module):
     """Normalize over d_model, using population variance and learned scale/shift."""
     def __init__(self, d_model: int, eps: float = 1e-5):
         super().__init__()
+        self.norm = nn.LayerNorm(d_model, eps=eps)
 
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
```

## Model: NMT interface documentation

File: `transformer/model/nmt.py`

The model docstrings now describe the implemented encoder and decoder interfaces rather than an unfinished backend. They clarify the source mask shape and explain that decoding returns hidden features before output projection. This feature changes documentation only and leaves the model computation unchanged.

### Before

```diff
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
 
-    This scaffold intentionally cannot train a real Transformer yet. Implement
-    components in model/ without nn.Transformer/MultiheadAttention shortcuts.
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
-        """src=(B,S), lengths=(B,) including EOS -> memory=(B,S,d_model)."""
         src_embeddings = self.positions(self.token_embeddings(src), src.eq(PAD_ID))
         encoded = self.encoder(src_embeddings, src_mask)
         return encoded
 
     def decode(self, tgt_in: Tensor, memory: Tensor, source_mask: Tensor) -> Tensor:
         """tgt_in=(B,T), memory=(B,S,d_model),
             source_mask=(B,1,1,S), True at source PAD; 
-            return logits (B,T,vocab_size)."""
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
```

### After

```diff
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
 
+    Components implement the pinned Multi30k text-only reference without
+    nn.Transformer/MultiheadAttention shortcuts.
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
+        """src=(B,S), src_mask=(B,1,1,S) -> memory=(B,S,d_model)."""
         src_embeddings = self.positions(self.token_embeddings(src), src.eq(PAD_ID))
         encoded = self.encoder(src_embeddings, src_mask)
         return encoded
 
     def decode(self, tgt_in: Tensor, memory: Tensor, source_mask: Tensor) -> Tensor:
         """tgt_in=(B,T), memory=(B,S,d_model),
             source_mask=(B,1,1,S), True at source PAD; 
+            return decoder features (B,T,d_model)."""
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
```

## Data: reference dictionary counts

File: `transformer/data/vocab.py`

Vocabulary construction materializes the training pairs so the corpus can be counted and reused consistently. Special-token counts include literal occurrences, and the end-of-sentence count includes the boundaries appended to both source and target sentences. This matches the reference dictionary counts as well as its token indices for the released training corpus.

### Before

```diff
 """Joined Fairseq-style dictionary: frequency order, lexical ties, padding to eight."""
 from collections import Counter
 from dataclasses import dataclass
 
 BOS_ID, PAD_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
 SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]
 
 
 @dataclass
 class Vocab:
     tokens: list[str]
     counts: list[int]
 
     def __post_init__(self):
         if self.tokens[:4] != SPECIALS or len(self.tokens) != len(self.counts):
             raise ValueError("invalid vocabulary framing")
         if len(set(self.tokens)) != len(self.tokens):
             raise ValueError("duplicate vocabulary tokens")
         self.indices = {token: i for i, token in enumerate(self.tokens)}
 
     def __len__(self):
         return len(self.tokens)
 
     def encode(self, tokens):
         return [self.indices.get(token, UNK_ID) for token in tokens]
 
     def decode(self, ids, *, reference=False):
         result = []
         for value in ids:
             value = int(value)
             if value == EOS_ID:
                 break
             if value in (PAD_ID, BOS_ID):
                 continue
             result.append("<<unk>>" if reference and value == UNK_ID else self.tokens[value])
         return result
 
     def to_dict(self):
         return {"tokens": self.tokens, "counts": self.counts}
 
 
 def build_vocab(pairs, padding_factor=8):
     if padding_factor < 1:
         raise ValueError("padding_factor must be positive")
     counts = Counter(token for src, tgt in pairs for row in (src, tgt) for token in row)
     for token in SPECIALS:
         counts.pop(token, None)
     ordered = sorted(counts, key=lambda token: (-counts[token], token))
-    tokens, frequencies = SPECIALS + ordered, [1] * 4 + [counts[t] for t in ordered]
     i = 0
     while len(tokens) % padding_factor:
         token = f"madeupword{i:04d}"
         i += 1
         if token not in counts:
             tokens.append(token)
             frequencies.append(0)
     return Vocab(tokens, frequencies)
```

### After

```diff
 """Joined Fairseq-style dictionary: frequency order, lexical ties, padding to eight."""
 from collections import Counter
 from dataclasses import dataclass
 
 BOS_ID, PAD_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
 SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]
 
 
 @dataclass
 class Vocab:
     tokens: list[str]
     counts: list[int]
 
     def __post_init__(self):
         if self.tokens[:4] != SPECIALS or len(self.tokens) != len(self.counts):
             raise ValueError("invalid vocabulary framing")
         if len(set(self.tokens)) != len(self.tokens):
             raise ValueError("duplicate vocabulary tokens")
         self.indices = {token: i for i, token in enumerate(self.tokens)}
 
     def __len__(self):
         return len(self.tokens)
 
     def encode(self, tokens):
         return [self.indices.get(token, UNK_ID) for token in tokens]
 
     def decode(self, ids, *, reference=False):
         result = []
         for value in ids:
             value = int(value)
             if value == EOS_ID:
                 break
             if value in (PAD_ID, BOS_ID):
                 continue
             result.append("<<unk>>" if reference and value == UNK_ID else self.tokens[value])
         return result
 
     def to_dict(self):
         return {"tokens": self.tokens, "counts": self.counts}
 
 
 def build_vocab(pairs, padding_factor=8):
     if padding_factor < 1:
         raise ValueError("padding_factor must be positive")
+    pairs = list(pairs)
     counts = Counter(token for src, tgt in pairs for row in (src, tgt) for token in row)
+    special_counts = [1 + counts.get(token, 0) for token in SPECIALS]
+    special_counts[EOS_ID] += 2 * len(pairs)
     for token in SPECIALS:
         counts.pop(token, None)
     ordered = sorted(counts, key=lambda token: (-counts[token], token))
+    tokens, frequencies = SPECIALS + ordered, special_counts + [counts[t] for t in ordered]
     i = 0
     while len(tokens) % padding_factor:
         token = f"madeupword{i:04d}"
         i += 1
         if token not in counts:
             tokens.append(token)
             frequencies.append(0)
     return Vocab(tokens, frequencies)
```

## Data: reference batch ordering and collation

File: `transformer/data/dataset.py`

The sampler now follows the reference permutation, stable length sorting and epoch-dependent batch shuffle. Token-budget batches prefer multiples of eight, and collation orders examples by descending source length while keeping each source paired with its target. These details matter because changing batch membership or ordering changes the gradients and dropout draws encountered during training.

### Before

```diff
 """Left-padded sources, right-padded targets, and length-grouped token-budget batches."""
 from dataclasses import dataclass
 import torch
 from torch.utils.data import Dataset, DataLoader, Sampler
 from transformer.data.vocab import EOS_ID, PAD_ID, Vocab
 
 
 class ParallelDataset(Dataset):
     def __init__(self, pairs, vocab: Vocab, max_positions=1024):
         self.pairs, self.vocab = pairs, vocab
         if not pairs:
             raise ValueError("empty dataset")
         self.rows = []
         for src, tgt in pairs:
             if not src or not tgt:
                 raise ValueError("empty content row")
             if max(len(src), len(tgt)) + 1 > max_positions:
                 raise ValueError("sequence exceeds max_positions; no silent truncation")
             self.rows.append((vocab.encode(src) + [EOS_ID], vocab.encode(tgt)))
 
     def __len__(self):
         return len(self.rows)
 
     def __getitem__(self, index):
         src, tgt = self.rows[index]
         return {"src": torch.tensor(src), "tgt_in": torch.tensor([EOS_ID, *tgt]),
                 "tgt_out": torch.tensor([*tgt, EOS_ID])}
 
 
 def pad_collate(rows):
     if not rows:
         raise ValueError("empty batch")
     b, s, t = len(rows), max(len(r["src"]) for r in rows), max(len(r["tgt_in"]) for r in rows)
     batch = {"src": torch.full((b, s), PAD_ID, dtype=torch.long),
              "tgt_in": torch.full((b, t), PAD_ID, dtype=torch.long),
              "tgt_out": torch.full((b, t), PAD_ID, dtype=torch.long),
              "src_lengths": torch.tensor([len(r["src"]) for r in rows])}
     for i, row in enumerate(rows):
         batch["src"][i, -len(row["src"]):] = row["src"]
         for name in ("tgt_in", "tgt_out"):
             batch[name][i, :len(row[name])] = row[name]
     return batch
 
 
 class TokenBatchSampler(Sampler):
-    def __init__(self, dataset, max_tokens=4096, *, shuffle=True, seed=1, batch_size=None):
         self.dataset, self.max_tokens = dataset, max_tokens
         self.shuffle, self.seed, self.batch_size = shuffle, seed, batch_size
         if max_tokens < 1 or (batch_size is not None and batch_size < 1):
             raise ValueError("batch limits must be positive")
 
     def __iter__(self):
-        generator = torch.Generator().manual_seed(self.seed)
-        indices = (torch.randperm(len(self.dataset), generator=generator).tolist()
-                   if self.shuffle else list(range(len(self.dataset))))
-        if self.shuffle:
-            indices.sort(key=lambda i: max(len(self.dataset.rows[i][0]), len(self.dataset.rows[i][1]) + 1))
-        batches, current, width = [], [], 0
-        for i in indices:
-            src, tgt = self.dataset.rows[i]
-            size = max(len(src), len(tgt) + 1)
             if size > self.max_tokens:
                 raise ValueError("one example exceeds token budget")
-            if current and ((max(width, size) * (len(current) + 1) > self.max_tokens)
                             or (self.batch_size is not None and len(current) >= self.batch_size)):
-                batches.append(current)
-                current, width = [], 0
             current.append(i)
-            width = max(width, size)
         if current:
             batches.append(current)
         if self.shuffle:
-            batches = [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]
         yield from batches
 
     def __len__(self):
         return sum(1 for _ in self)
 
 
 @dataclass
 class ProcessedParallel:
     dataset: ParallelDataset
     vocab: Vocab
 
-    def dataloader(self, max_tokens=4096, *, shuffle=False, seed=1, batch_size=None):
         return DataLoader(self.dataset, batch_sampler=TokenBatchSampler(
-            self.dataset, max_tokens, shuffle=shuffle, seed=seed, batch_size=batch_size),
-            collate_fn=pad_collate, generator=torch.Generator().manual_seed(seed))
 
 
 def process_pairs(pairs, vocab, max_positions=1024):
     return ProcessedParallel(ParallelDataset(pairs, vocab, max_positions), vocab)
```

### After

```diff
 """Left-padded sources, right-padded targets, and length-grouped token-budget batches."""
 from dataclasses import dataclass
 import torch
+import numpy as np
 from torch.utils.data import Dataset, DataLoader, Sampler
 from transformer.data.vocab import EOS_ID, PAD_ID, Vocab
 
 
 class ParallelDataset(Dataset):
     def __init__(self, pairs, vocab: Vocab, max_positions=1024):
         self.pairs, self.vocab = pairs, vocab
         if not pairs:
             raise ValueError("empty dataset")
         self.rows = []
         for src, tgt in pairs:
             if not src or not tgt:
                 raise ValueError("empty content row")
             if max(len(src), len(tgt)) + 1 > max_positions:
                 raise ValueError("sequence exceeds max_positions; no silent truncation")
             self.rows.append((vocab.encode(src) + [EOS_ID], vocab.encode(tgt)))
 
     def __len__(self):
         return len(self.rows)
 
     def __getitem__(self, index):
         src, tgt = self.rows[index]
         return {"src": torch.tensor(src), "tgt_in": torch.tensor([EOS_ID, *tgt]),
                 "tgt_out": torch.tensor([*tgt, EOS_ID])}
 
 
 def pad_collate(rows):
     if not rows:
         raise ValueError("empty batch")
     b, s, t = len(rows), max(len(r["src"]) for r in rows), max(len(r["tgt_in"]) for r in rows)
     batch = {"src": torch.full((b, s), PAD_ID, dtype=torch.long),
              "tgt_in": torch.full((b, t), PAD_ID, dtype=torch.long),
              "tgt_out": torch.full((b, t), PAD_ID, dtype=torch.long),
              "src_lengths": torch.tensor([len(r["src"]) for r in rows])}
     for i, row in enumerate(rows):
         batch["src"][i, -len(row["src"]):] = row["src"]
         for name in ("tgt_in", "tgt_out"):
             batch[name][i, :len(row[name])] = row[name]
     return batch
 
 
+def reference_collate(rows):
+    """Fairseq sorts each mini-batch by descending source length."""
+    return pad_collate(sorted(rows, key=lambda row: len(row['src']), reverse=True))
+
+
 class TokenBatchSampler(Sampler):
+    """Pinned Fairseq stable length ordering, batch multiples, and epoch shuffling."""
+    def __init__(self, dataset, max_tokens=4096, *, shuffle=True, seed=1,
+                 batch_size=None, epoch=1, required_multiple=8):
         self.dataset, self.max_tokens = dataset, max_tokens
         self.shuffle, self.seed, self.batch_size = shuffle, seed, batch_size
+        self.epoch, self.required_multiple = epoch, required_multiple
         if max_tokens < 1 or (batch_size is not None and batch_size < 1):
             raise ValueError("batch limits must be positive")
 
     def __iter__(self):
+        rng = np.random.RandomState(self.seed)
+        indices = rng.permutation(len(self.dataset)) if self.shuffle else np.arange(len(self.dataset))
+        src = np.array([len(r[0]) for r in self.dataset.rows])
+        tgt = np.array([len(r[1]) + 1 for r in self.dataset.rows])
+        indices = indices[np.argsort(tgt[indices], kind='mergesort')]
+        indices = indices[np.argsort(src[indices], kind='mergesort')]
+        batches, current, sizes = [], [], []
+        for index in indices:
+            i = int(index)
+            size = int(max(src[i], tgt[i]))
             if size > self.max_tokens:
                 raise ValueError("one example exceeds token budget")
+            sizes.append(size)
+            if current and (max(sizes) * (len(current) + 1) > self.max_tokens
                             or (self.batch_size is not None and len(current) >= self.batch_size)):
+                m = self.required_multiple
+                split = max(m * (len(current) // m), len(current) % m)
+                batches.append(current[:split])
+                current, sizes = current[split:], sizes[split:]
             current.append(i)
         if current:
             batches.append(current)
         if self.shuffle:
+            order = np.random.RandomState(self.seed + self.epoch).permutation(len(batches))
+            batches = [batches[i] for i in order]
         yield from batches
 
     def __len__(self):
         return sum(1 for _ in self)
 
 
 @dataclass
 class ProcessedParallel:
     dataset: ParallelDataset
     vocab: Vocab
 
+    def dataloader(self, max_tokens=4096, *, shuffle=False, seed=1, batch_size=None, epoch=1):
         return DataLoader(self.dataset, batch_sampler=TokenBatchSampler(
+            self.dataset, max_tokens, shuffle=shuffle, seed=seed, batch_size=batch_size, epoch=epoch),
+            collate_fn=reference_collate, generator=torch.Generator().manual_seed(seed))
 
 
 def process_pairs(pairs, vocab, max_positions=1024):
     return ProcessedParallel(ParallelDataset(pairs, vocab, max_positions), vocab)
```

## Training: reference Adam arithmetic

File: `transformer/training/adam.py`

A dedicated Adam optimizer reproduces the arithmetic used by the pinned reference implementation. It adds epsilon to the square root of the uncorrected second moment and incorporates bias correction into the step size. This avoids the numerical difference introduced by using the current PyTorch Adam convention with the same nominal hyperparameters.

### Before

New file: there was no before code.

### After

```diff
+"""Dense Adam matching the pinned Fairseq epsilon/bias-correction convention."""
+import math
+import torch
+
+
+class ReferenceAdam(torch.optim.Optimizer):
+    def __init__(self, params, lr, betas=(.9, .98), eps=1e-8):
+        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))
+
+    @torch.no_grad()
+    def step(self, closure=None):
+        loss = None
+        if closure is not None:
+            with torch.enable_grad():
+                loss = closure()
+        for group in self.param_groups:
+            beta1, beta2 = group['betas']
+            for p in group['params']:
+                if p.grad is None:
+                    continue
+                grad = p.grad
+                if grad.is_sparse:
+                    raise RuntimeError('ReferenceAdam requires dense gradients')
+                state = self.state[p]
+                if not state:
+                    state.update(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
+                state['step'] += 1
+                m, v = state['exp_avg'], state['exp_avg_sq']
+                m.mul_(beta1).add_(grad, alpha=1 - beta1)
+                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
+                correction = math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])
+                p.addcdiv_(m, v.sqrt().add_(group['eps']), value=-group['lr'] * correction)
+        return loss
```

## Training: optimizer integration, seeding, epoch batching and checkpoint retention

File: `transformer/training/loop.py`

The training loop now uses the reference Adam optimizer, passes epoch information to the sampler and reseeds dropout at each optimizer update. Two accumulated microbatches approximate the reference two-GPU effective batch, although dropout streams and gradient reduction order can still differ. The loop logs progress and retains the last ten complete epoch checkpoints, keeping a terminal partial epoch separate from checkpoint averaging.

### Before

```diff
 """Token-normalized accumulation, reference Adam schedule, and resumable epochs."""
 from dataclasses import asdict
 import itertools
 from pathlib import Path
 import torch
 from transformer.config import TransformerConfig
 from transformer.training.checkpoints import load_checkpoint, save_checkpoint
 from transformer.training.loss import token_loss
 from transformer.training.schedule import learning_rate
 from transformer.eval.metrics import evaluate_perplexity
 
 
 def train_update(model, batches, optimizer, config, device):
     """One optimizer update from summed token losses across unequal microbatches."""
     optimizer.zero_grad(set_to_none=True)
     total_loss = total_nll = total_tokens = 0
     for batch in batches:
         batch = {k: v.to(device) for k, v in batch.items()}
         logits = model(batch["src"], batch["src_lengths"], batch["tgt_in"])
         loss, nll, count = token_loss(logits, batch["tgt_out"], config.label_smoothing)
         loss.backward()
         total_loss += float(loss.detach())
         total_nll += float(nll.detach())
         total_tokens += count
     if not total_tokens:
         raise ValueError("optimizer update has no valid target tokens")
     for parameter in model.parameters():
         if parameter.grad is not None:
             parameter.grad.div_(total_tokens)
     norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
     optimizer.step()
     return {"loss": total_loss / total_tokens, "nll": total_nll / total_tokens,
             "tokens": total_tokens, "grad_norm": float(norm)}
 
 
 def fit(model, train, valid, config: TransformerConfig, device, *, output_dir,
         metadata=None, resume=None, verbose=True):
     config.validate()
     model.to(device)
-    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(0, config),
                                  betas=config.adam_betas, eps=config.adam_eps)
     metadata = dict(metadata or {})
     metadata.setdefault("config", asdict(config))
     metadata.setdefault("vocab", train.vocab.to_dict())
     state = {"epoch": 1, "next_batch": 0, "updates": 0,
              "best_loss": float("inf"), "bad_epochs": 0, "history": []}
     if resume is not None:
         saved = load_checkpoint(resume)
         old_config = dict(saved["metadata"]["config"])
         new_config = asdict(config)
         for limits in ("max_updates", "max_epochs"):
             old_config.pop(limits, None)
             new_config.pop(limits, None)
         if old_config != new_config or saved["metadata"]["vocab"] != train.vocab.to_dict():
             raise ValueError("resume configuration/vocabulary mismatch")
         for key in ("data_manifest", "code_sha256"):
             if saved["metadata"].get(key) != metadata.get(key):
                 raise ValueError(f"resume {key} mismatch")
         load_checkpoint(resume, model=model, optimizer=optimizer, restore_rng=True)
         state = saved["state"]
         metadata = {**saved["metadata"], "config": asdict(config)}
     output_dir = Path(output_dir)
     while (state["epoch"] <= config.max_epochs and state["updates"] < config.max_updates
            and state["bad_epochs"] < config.patience):
         epoch = state["epoch"]
         model.train()
-        loader = train.dataloader(config.max_tokens, shuffle=True, seed=config.seed + epoch,
-                                  batch_size=config.batch_size)
         total_batches = len(loader)
         iterator = itertools.islice(iter(loader), state["next_batch"], None)
         while group := list(itertools.islice(iterator, config.accumulation_steps)):
             for parameters in optimizer.param_groups:
                 parameters["lr"] = learning_rate(state["updates"], config)
             train_update(model, group, optimizer, config, device)
             state["updates"] += 1
             state["next_batch"] += len(group)
             if state["updates"] >= config.max_updates:
                 break
         metrics = evaluate_perplexity(model, valid, device, max_tokens=config.max_tokens,
                                       batch_size=config.batch_size, epsilon=config.label_smoothing)
         completed = state["next_batch"] >= total_batches
         if completed:
             state["bad_epochs"] = 0 if metrics["mean_loss"] < state["best_loss"] else state["bad_epochs"] + 1
             state["best_loss"] = min(state["best_loss"], metrics["mean_loss"])
             state["epoch"], state["next_batch"] = epoch + 1, 0
         state["history"].append({"epoch": epoch, "updates": state["updates"],
                                  "epoch_complete": completed, "validation": metrics})
         filename = f"checkpoint_epoch_{epoch:04d}.pt" if completed else "checkpoint_partial.pt"
         save_checkpoint(output_dir / filename, model, optimizer, state, metadata)
         if verbose:
             print(f"epoch={epoch} updates={state['updates']} valid_ppl={metrics['perplexity']:.4f} complete={completed}")
     return state
```

### After

```diff
 """Token-normalized accumulation, reference Adam schedule, and resumable epochs."""
 from dataclasses import asdict
 import itertools
 from pathlib import Path
 import torch
 from transformer.config import TransformerConfig
 from transformer.training.checkpoints import load_checkpoint, save_checkpoint
 from transformer.training.loss import token_loss
 from transformer.training.schedule import learning_rate
+from transformer.training.adam import ReferenceAdam
 from transformer.eval.metrics import evaluate_perplexity
 
 
 def train_update(model, batches, optimizer, config, device):
     """One optimizer update from summed token losses across unequal microbatches."""
     optimizer.zero_grad(set_to_none=True)
     total_loss = total_nll = total_tokens = 0
     for batch in batches:
         batch = {k: v.to(device) for k, v in batch.items()}
         logits = model(batch["src"], batch["src_lengths"], batch["tgt_in"])
         loss, nll, count = token_loss(logits, batch["tgt_out"], config.label_smoothing)
         loss.backward()
         total_loss += float(loss.detach())
         total_nll += float(nll.detach())
         total_tokens += count
     if not total_tokens:
         raise ValueError("optimizer update has no valid target tokens")
     for parameter in model.parameters():
         if parameter.grad is not None:
             parameter.grad.div_(total_tokens)
     norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
     optimizer.step()
     return {"loss": total_loss / total_tokens, "nll": total_nll / total_tokens,
             "tokens": total_tokens, "grad_norm": float(norm)}
 
 
 def fit(model, train, valid, config: TransformerConfig, device, *, output_dir,
         metadata=None, resume=None, verbose=True):
     config.validate()
     model.to(device)
+    optimizer = ReferenceAdam(model.parameters(), lr=learning_rate(0, config),
                                  betas=config.adam_betas, eps=config.adam_eps)
     metadata = dict(metadata or {})
     metadata.setdefault("config", asdict(config))
     metadata.setdefault("vocab", train.vocab.to_dict())
     state = {"epoch": 1, "next_batch": 0, "updates": 0,
              "best_loss": float("inf"), "bad_epochs": 0, "history": []}
     if resume is not None:
         saved = load_checkpoint(resume)
         old_config = dict(saved["metadata"]["config"])
         new_config = asdict(config)
         for limits in ("max_updates", "max_epochs"):
             old_config.pop(limits, None)
             new_config.pop(limits, None)
         if old_config != new_config or saved["metadata"]["vocab"] != train.vocab.to_dict():
             raise ValueError("resume configuration/vocabulary mismatch")
         for key in ("data_manifest", "code_sha256"):
             if saved["metadata"].get(key) != metadata.get(key):
                 raise ValueError(f"resume {key} mismatch")
         load_checkpoint(resume, model=model, optimizer=optimizer, restore_rng=True)
         state = saved["state"]
         metadata = {**saved["metadata"], "config": asdict(config)}
     output_dir = Path(output_dir)
     while (state["epoch"] <= config.max_epochs and state["updates"] < config.max_updates
            and state["bad_epochs"] < config.patience):
         epoch = state["epoch"]
         model.train()
+        loader = train.dataloader(config.max_tokens, shuffle=True, seed=config.seed,
+                                  batch_size=config.batch_size, epoch=epoch)
         total_batches = len(loader)
         iterator = itertools.islice(iter(loader), state["next_batch"], None)
         while group := list(itertools.islice(iterator, config.accumulation_steps)):
+            # The reference resets dropout RNG at each update; resume is reproducible.
+            torch.manual_seed(config.seed + state['updates'])
             for parameters in optimizer.param_groups:
                 parameters["lr"] = learning_rate(state["updates"], config)
             train_update(model, group, optimizer, config, device)
             state["updates"] += 1
             state["next_batch"] += len(group)
+            if verbose and state['updates'] % 100 == 0:
+                print(f"update={state['updates']} epoch={epoch} lr={optimizer.param_groups[0]['lr']:.7f}", flush=True)
             if state["updates"] >= config.max_updates:
                 break
         metrics = evaluate_perplexity(model, valid, device, max_tokens=config.max_tokens,
                                       batch_size=config.batch_size, epsilon=config.label_smoothing)
         completed = state["next_batch"] >= total_batches
         if completed:
             state["bad_epochs"] = 0 if metrics["mean_loss"] < state["best_loss"] else state["bad_epochs"] + 1
             state["best_loss"] = min(state["best_loss"], metrics["mean_loss"])
             state["epoch"], state["next_batch"] = epoch + 1, 0
         state["history"].append({"epoch": epoch, "updates": state["updates"],
                                  "epoch_complete": completed, "validation": metrics})
         filename = f"checkpoint_epoch_{epoch:04d}.pt" if completed else "checkpoint_partial.pt"
         save_checkpoint(output_dir / filename, model, optimizer, state, metadata)
+        if completed:
+            # Match retention of the last ten complete epoch checkpoints.
+            for old in sorted(output_dir.glob('checkpoint_epoch_*.pt'))[:-10]:
+                old.unlink()
         if verbose:
             print(f"epoch={epoch} updates={state['updates']} valid_ppl={metrics['perplexity']:.4f} complete={completed}")
     return state
```

## Training: output protection and provenance

File: `scripts/transformer/train/run.py`

The training runner now rejects a nonempty output directory unless an explicit resume checkpoint is supplied. It removes an obsolete constructor check that expected the model to be an unimplemented scaffold. Run metadata records the pinned reference revision and the single-GPU accumulation difference so results retain their experimental context.

### Before

```diff
 """Usage: python train.py transformer --smoke | --profile multi30k_tiny."""
 import argparse
 from dataclasses import asdict
 import hashlib
 import json
 from pathlib import Path
 import random
 import subprocess
 import torch
 from scripts.paths import DATA_DIR, REPO_ROOT
 from transformer.config import TransformerConfig
 from transformer.data.corpus import load_pairs, smoke_pairs
 from transformer.data.dataset import process_pairs
 from transformer.data.download import download_multi30k
 from transformer.data.vocab import build_vocab
 from transformer.model.nmt import TransformerNMT
 from transformer.training.loop import fit
 from transformer.training.smoke import DummyTransformer
 
 
 def code_provenance():
     roots = (REPO_ROOT / "transformer", REPO_ROOT / "scripts/transformer")
     hashes = {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for root in roots for path in sorted(root.rglob("*.py"))}
     return {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
             "code_sha256": hashes}
 
 
 def main(argv=None):
     parser = argparse.ArgumentParser(description=__doc__)
     parser.add_argument("--smoke", action="store_true")
     parser.add_argument("--profile", choices=["multi30k_tiny"], default="multi30k_tiny")
     parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
     parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".tmp/transformer/runs")
     parser.add_argument("--resume", type=Path)
     parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
     parser.add_argument("--threads", type=int, default=1)
     args = parser.parse_args(argv)
     config = TransformerConfig.smoke() if args.smoke else TransformerConfig()
     config.validate()
     if args.threads < 1:
         parser.error("threads must be positive")
     torch.set_num_threads(args.threads)
     torch.manual_seed(config.seed)
     random.seed(config.seed)
     chosen = torch.device(args.device)
     provenance = code_provenance()
     if args.smoke:
         pairs = smoke_pairs()
         train_pairs, valid_pairs = pairs, pairs
         provenance["data_manifest"] = {"smoke": True, "evaluation_split": "same offline fixture; infrastructure only"}
     else:
-        # Fail clearly before a potentially unnecessary dataset download.
-        try:
-            TransformerNMT(4, config)
-        except NotImplementedError as error:
-            parser.exit(2, f"{error}. Use --smoke for infrastructure verification.\n")
         directory = download_multi30k(args.data_dir)
         train_pairs = load_pairs(directory / "train.en", directory / "train.de")
         valid_pairs = load_pairs(directory / "valid.en", directory / "valid.de")
         provenance["data_manifest"] = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
     vocab = build_vocab(train_pairs)
     train, valid = (process_pairs(p, vocab, config.max_positions) for p in (train_pairs, valid_pairs))
     model = (DummyTransformer(len(vocab), config.d_model) if args.smoke else TransformerNMT(len(vocab), config))
     metadata = {**provenance, "config": asdict(config), "vocab": vocab.to_dict(),
                 "model_kind": "dummy_infrastructure" if args.smoke else "transformer",
-                "reference_bleu": None if args.smoke else 41.02}
     state = fit(model, train, valid, config, chosen, output_dir=args.output_dir, metadata=metadata, resume=args.resume)
     report = {"kind": metadata["model_kind"], "training_state": state,
               "notice": "Dummy metrics verify infrastructure only; not Transformer results." if args.smoke else ""}
     args.output_dir.mkdir(parents=True, exist_ok=True)
     (args.output_dir / "training-results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
     print(json.dumps(report, indent=2))
     return report
 
 
 if __name__ == "__main__":
     main()
```

### After

```diff
 """Usage: python train.py transformer --smoke | --profile multi30k_tiny."""
 import argparse
 from dataclasses import asdict
 import hashlib
 import json
 from pathlib import Path
 import random
 import subprocess
 import torch
 from scripts.paths import DATA_DIR, REPO_ROOT
 from transformer.config import TransformerConfig
 from transformer.data.corpus import load_pairs, smoke_pairs
 from transformer.data.dataset import process_pairs
 from transformer.data.download import download_multi30k
 from transformer.data.vocab import build_vocab
 from transformer.model.nmt import TransformerNMT
 from transformer.training.loop import fit
 from transformer.training.smoke import DummyTransformer
 
 
 def code_provenance():
     roots = (REPO_ROOT / "transformer", REPO_ROOT / "scripts/transformer")
     hashes = {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for root in roots for path in sorted(root.rglob("*.py"))}
     return {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
             "code_sha256": hashes}
 
 
 def main(argv=None):
     parser = argparse.ArgumentParser(description=__doc__)
     parser.add_argument("--smoke", action="store_true")
     parser.add_argument("--profile", choices=["multi30k_tiny"], default="multi30k_tiny")
     parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
     parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".tmp/transformer/runs")
     parser.add_argument("--resume", type=Path)
     parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
     parser.add_argument("--threads", type=int, default=1)
     args = parser.parse_args(argv)
+    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
+        parser.error('output directory is not empty; choose a fresh directory or use --resume')
     config = TransformerConfig.smoke() if args.smoke else TransformerConfig()
     config.validate()
     if args.threads < 1:
         parser.error("threads must be positive")
     torch.set_num_threads(args.threads)
     torch.manual_seed(config.seed)
     random.seed(config.seed)
     chosen = torch.device(args.device)
     provenance = code_provenance()
     if args.smoke:
         pairs = smoke_pairs()
         train_pairs, valid_pairs = pairs, pairs
         provenance["data_manifest"] = {"smoke": True, "evaluation_split": "same offline fixture; infrastructure only"}
     else:
         directory = download_multi30k(args.data_dir)
         train_pairs = load_pairs(directory / "train.en", directory / "train.de")
         valid_pairs = load_pairs(directory / "valid.en", directory / "valid.de")
         provenance["data_manifest"] = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
     vocab = build_vocab(train_pairs)
     train, valid = (process_pairs(p, vocab, config.max_positions) for p in (train_pairs, valid_pairs))
     model = (DummyTransformer(len(vocab), config.d_model) if args.smoke else TransformerNMT(len(vocab), config))
     metadata = {**provenance, "config": asdict(config), "vocab": vocab.to_dict(),
                 "model_kind": "dummy_infrastructure" if args.smoke else "transformer",
+              "reference_bleu": None if args.smoke else 41.02,
+              "reference_commit": "dc368d0af8d60270b8f4aaa2f3ac58771c551da3",
+              "runtime_difference": "Single GPU with two microbatches per update instead of two-GPU DDP."}
     state = fit(model, train, valid, config, chosen, output_dir=args.output_dir, metadata=metadata, resume=args.resume)
     report = {"kind": metadata["model_kind"], "training_state": state,
               "notice": "Dummy metrics verify infrastructure only; not Transformer results." if args.smoke else ""}
     args.output_dir.mkdir(parents=True, exist_ok=True)
     (args.output_dir / "training-results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
     print(json.dumps(report, indent=2))
     return report
 
 
 if __name__ == "__main__":
     main()
```

## Evaluation: batched beam search, encoder reuse and token rules

File: `transformer/eval/decoding.py`

Beam search now processes hypotheses in batches and reuses encoder memory across their decoder calls. Candidate filtering, minimum length, forced end-of-sentence termination and length normalization follow the reference rules. Decoder prefixes are still recomputed without an incremental key/value cache, so this follows the same mathematical approach with a remaining computational and floating-point difference.

### Before

```diff
-"""Uncached autoregressive greedy/beam decoding through the forward contract."""
 import torch
-from transformer.data.vocab import BOS_ID, PAD_ID, EOS_ID
 
 
 @torch.no_grad()
 def generate(model, src, src_lengths, *, beam_size=5, max_tokens=200, length_penalty=1.):
-    """Return (B,L) ids excluding decoder-start EOS, padded after generated EOS.
-
-    Use top-2*beam candidates, length-normalized finished scores, no sampling,
-    no repeated-ngram blocking. Reference limit: up to max_tokens content
-    steps plus one forced EOS step. Minimum content length is one.
-    """
     if beam_size < 1 or max_tokens < 1 or length_penalty < 0:
         raise ValueError("invalid decoding limits")
     model.eval()
-    outputs = []
-    for row in range(src.size(0)):
-        active, finished = [([EOS_ID], 0.)], []
-        for step in range(max_tokens + 1):
-            prefixes = torch.tensor([tokens for tokens, _ in active], device=src.device)
-            logits = model(src[row:row + 1].expand(len(active), -1),
-                           src_lengths[row:row + 1].expand(len(active)), prefixes)[:, -1]
-            scores = logits.log_softmax(-1)
-            scores[:, PAD_ID] = -float("inf")
-            scores[:, BOS_ID] = -float("inf")
-            if step == 0:
-                scores[:, EOS_ID] = -float("inf")
-            if step == max_tokens:
-                eos_scores = scores[:, EOS_ID].clone()
-                scores.fill_(-float("inf"))
-                scores[:, EOS_ID] = eos_scores
-            scores += scores.new_tensor([s for _, s in active])[:, None]
-            top_scores, top_indices = scores.flatten().topk(min(2 * beam_size, scores.numel()))
-            next_active = []
-            for rank, (score, index) in enumerate(zip(top_scores.tolist(), top_indices.tolist())):
-                if score == -float("inf"):
-                    continue
-                parent, token = divmod(index, scores.size(1))
-                tokens = [*active[parent][0], token]
-                if token == EOS_ID:
-                    if rank < beam_size:
-                        finished.append((tokens[1:], score / ((len(tokens) - 1) ** length_penalty)))
-                elif len(next_active) < beam_size:
-                    next_active.append((tokens, score))
-            finished.sort(key=lambda item: item[1], reverse=True)
-            finished = finished[:beam_size]
-            if len(finished) >= beam_size or not next_active:
-                break
-            active = next_active
-        if not finished:
-            raise RuntimeError("decoder could not produce EOS within its configured limit")
-        outputs.append(finished[0][0])
-    width = max(map(len, outputs))
-    result = src.new_full((len(outputs), width), PAD_ID)
-    for row, values in enumerate(outputs):
-        result[row, :len(values)] = src.new_tensor(values)
     return result
```

### After

```diff
+"""Batched, uncached beam search matching the reference's score and EOS rules.
+
+Encoder memory is reused. Target prefixes are recomputed; no incremental cache.
+"""
 import torch
+from transformer.data.vocab import PAD_ID, EOS_ID
 
 
 @torch.no_grad()
 def generate(model, src, src_lengths, *, beam_size=5, max_tokens=200, length_penalty=1.):
     if beam_size < 1 or max_tokens < 1 or length_penalty < 0:
         raise ValueError("invalid decoding limits")
     model.eval()
+    batch = src.size(0)
+    h = beam_size
+    prefixes = src.new_full((batch, h, 1), EOS_ID)
+    cumulative = torch.full((batch, h), -float('inf'), device=src.device)
+    cumulative[:, 0] = 0.
+    finished = [[] for _ in range(batch)]
+    done = [False] * batch
+    memory = source_mask = None
+    if hasattr(model, 'encode') and hasattr(model, 'decode') and hasattr(model, 'output_projection'):
+        source_mask = src.eq(PAD_ID)[:, None, None, :]
+        memory = model.encode(src, source_mask).repeat_interleave(h, dim=0)
+        source_mask = source_mask.repeat_interleave(h, dim=0)
+    expanded_src = src.repeat_interleave(h, dim=0)
+    expanded_lengths = src_lengths.repeat_interleave(h, dim=0)
+    for step in range(max_tokens + 1):
+        flat = prefixes.reshape(batch * h, -1)
+        if memory is None:
+            logits = model(expanded_src, expanded_lengths, flat)[:, -1]
+        else:
+            features = model.decode(flat, memory, source_mask)
+            logits = model.output_projection(features[:, -1])
+        log_probs = logits.float().log_softmax(-1).view(batch, h, -1)
+        vocab = log_probs.size(-1)
+        log_probs[:, :, PAD_ID] = -float('inf')
+        if step == 0:
+            log_probs[:, :, EOS_ID] = -float('inf')
+        if step == max_tokens:
+            eos = log_probs[:, :, EOS_ID].clone()
+            log_probs.fill_(-float('inf'))
+            log_probs[:, :, EOS_ID] = eos
+        scores = (log_probs + cumulative[:, :, None]).flatten(1)
+        top_scores, top_indices = scores.topk(min(2 * h, scores.size(1)), dim=1)
+        candidate_scores, candidate_ids = top_scores.tolist(), top_indices.tolist()
+        old_prefixes = prefixes.tolist()
+        next_prefixes, next_scores = [], []
+        for b in range(batch):
+            live, live_scores = [], []
+            if not done[b]:
+                for rank, (score, index) in enumerate(zip(candidate_scores[b], candidate_ids[b])):
+                    if score == -float('inf'):
+                        continue
+                    parent, token = divmod(index, vocab)
+                    sequence = old_prefixes[b][parent] + [token]
+                    if token == EOS_ID:
+                        if rank < h:
+                            finished[b].append((sequence[1:], score / ((step + 1) ** length_penalty)))
+                    elif len(live) < h:
+                        live.append(sequence)
+                        live_scores.append(score)
+                finished[b].sort(key=lambda item: item[1], reverse=True)
+                finished[b] = finished[b][:h]
+                done[b] = len(finished[b]) >= h or not live
+            while len(live) < h:
+                live.append([EOS_ID] * (step + 2))
+                live_scores.append(-float('inf'))
+            next_prefixes.append(live)
+            next_scores.append(live_scores)
+        if all(done):
+            break
+        prefixes = src.new_tensor(next_prefixes)
+        cumulative = torch.tensor(next_scores, device=src.device)
+    if any(not row for row in finished):
+        raise RuntimeError("decoder could not produce EOS within its configured limit")
+    best = [row[0][0] for row in finished]
+    result = src.new_full((batch, max(map(len, best))), PAD_ID)
+    for b, values in enumerate(best):
+        result[b, :len(values)] = src.new_tensor(values)
     return result
```

## Validation: integrated model instead of stub expectations

File: `tests/test_transformer_scaffolding.py`

The scaffold tests now exercise an implemented Transformer instead of expecting its model constructors to raise errors. An integrated forward and backward check verifies finite outputs, gradients and shared embedding weights. The evaluation ordering check now verifies example coverage while allowing the reference length-based batch order.

### Before

```diff
-"""Infrastructure checks; the real Transformer remains an intentional stub."""
 import copy
 from dataclasses import replace
 import importlib
 import json
 import math
 from pathlib import Path
 import tempfile
 import unittest
 from unittest.mock import patch
 import urllib.error
 import torch
 from torch import nn
 
 from scripts.dispatch import available_models
 from scripts.paths import REPO_ROOT
 from transformer.config import TransformerConfig
 from transformer.data.corpus import load_pairs, smoke_pairs
 from transformer.data.dataset import process_pairs, pad_collate, TokenBatchSampler
 from transformer.data.download import FILES, download_multi30k
 from transformer.data.masks import causal_mask, padding_mask, decoder_mask
 from transformer.data.vocab import build_vocab, PAD_ID, EOS_ID, UNK_ID
 from transformer.eval.bleu import corpus_bleu, remove_bpe
 from transformer.eval.decoding import generate
 from transformer.eval.metrics import evaluate_perplexity
 from transformer.model.attention import scaled_dot_product_attention
 from transformer.training.checkpoints import save_checkpoint, load_checkpoint, average_checkpoints
 from transformer.training.loop import fit, train_update
 from transformer.training.loss import token_loss
 from transformer.training.schedule import learning_rate
 from transformer.training.smoke import DummyTransformer
 
 
 class TransformerScaffoldingTests(unittest.TestCase):
     @classmethod
     def setUpClass(cls):
         torch.set_num_threads(1)
         cls.scratch = REPO_ROOT / ".tmp/transformer-scaffold-tests"
         cls.scratch.mkdir(parents=True, exist_ok=True)
 
     def temporary_directory(self):
         context = tempfile.TemporaryDirectory(dir=self.scratch)
         self.addCleanup(context.cleanup)
         return Path(context.name)
 
     def processed(self):
         pairs = smoke_pairs()
         return process_pairs(pairs, build_vocab(pairs))
 
     def test_configuration_validation_and_smoke_defaults(self):
         TransformerConfig().validate()
         TransformerConfig.smoke().validate()
         for overrides in ({"heads": 3}, {"max_tokens": 0}, {"seed": -1},
                           {"label_smoothing": 1.}, {"adam_eps": float("nan")},
                           {"warmup_init_lr": .1}):
             with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                 replace(TransformerConfig(), **overrides).validate()
 
     def test_joined_vocab_lexical_ties_and_no_eval_leakage(self):
         vocab = build_vocab([(["z", "a", "a"], ["b", "z"])])
         self.assertEqual(vocab.tokens[:8], ["<s>", "<pad>", "</s>", "<unk>", "a", "z", "b", "madeupword0000"])
         self.assertEqual(vocab.encode(["validation_only"]), [UNK_ID])
         self.assertEqual(len(vocab) % 8, 0)
 
     def test_source_eos_left_padding_target_shift_and_masking(self):
         pairs = [(["a"], ["x", "y"]), (["a", "b", "c"], ["x"])]
         processed = process_pairs(pairs, build_vocab(pairs))
         batch = pad_collate([processed.dataset[i] for i in (0, 1)])
         self.assertEqual(batch["src_lengths"].tolist(), [2, 4])
         self.assertEqual(batch["src"][0, :2].tolist(), [PAD_ID, PAD_ID])
         self.assertEqual(batch["src"][:, -1].tolist(), [EOS_ID, EOS_ID])
         self.assertEqual(batch["tgt_in"][:, 0].tolist(), [EOS_ID, EOS_ID])
         self.assertEqual(batch["tgt_out"][1, 1:].tolist(), [EOS_ID, PAD_ID])
         mask = decoder_mask(batch["tgt_in"])
         self.assertTrue(mask[0, 0, 0, 1])
         self.assertFalse(mask[0, 0, 1, 1])
         self.assertTrue(mask[1, 0, 2, 2])
         self.assertEqual(padding_mask(batch["src"])[0, 0, 0].tolist(), [True, True, False, False])
 
     def test_empty_and_overlong_rows_rejected_without_truncation(self):
         vocab = build_vocab(smoke_pairs())
         for pairs in ([([], ["x"])], [(["a"], [])], []):
             with self.assertRaises(ValueError):
                 process_pairs(pairs, vocab)
         with self.assertRaises(ValueError):
             process_pairs([(["a"] * 5, ["b"])], vocab, max_positions=5)
         self.assertTrue(decoder_mask(torch.full((1, 3), PAD_ID)).all())
         self.assertEqual(causal_mask(3)[0, 0].sum().item(), 3)
 
     def test_parallel_alignment_and_empty_content(self):
         folder = self.temporary_directory()
         src, tgt = folder / "src", folder / "tgt"
         src.write_text("one\ntwo\n", encoding="utf-8")
         tgt.write_text("eins\n", encoding="utf-8")
         with self.assertRaisesRegex(ValueError, "mismatch"):
             load_pairs(src, tgt)
         tgt.write_text("eins\n\n", encoding="utf-8")
         with self.assertRaisesRegex(ValueError, "empty"):
             load_pairs(src, tgt)
 
     def test_token_budget_covers_all_rows_and_preserves_eval_order(self):
         p = self.processed()
         sampler = TokenBatchSampler(p.dataset, 15, seed=11)
         batches = list(sampler)
         self.assertEqual(sorted(i for b in batches for i in b), list(range(4)))
         self.assertEqual(batches, list(sampler))
         for indices in batches:
             batch = pad_collate([p.dataset[i] for i in indices])
             self.assertLessEqual(max(batch["src"].size(1), batch["tgt_in"].size(1)) * len(indices), 15)
-        self.assertEqual([i for b in TokenBatchSampler(p.dataset, 15, shuffle=False) for i in b], list(range(4)))
         with self.assertRaises(ValueError):
             list(TokenBatchSampler(p.dataset, 1))
 
     def test_cached_download_checksums_and_counts(self):
         folder = self.temporary_directory()
         fixture = folder / "fixture"
         fixture.mkdir()
         for name in FILES:
             (fixture / name).write_text("#version: 0.2\na b\n" if name == "code" else "one two\n", encoding="utf-8")
         expected = {"train": 1, "valid": 1, "test.2016": 1}
         directory = download_multi30k(folder / "output", base_url=fixture.as_uri(), expected_splits=expected)
         with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
             download_multi30k(folder / "output", base_url=fixture.as_uri(), expected_splits=expected)
         manifest = json.loads((directory / "manifest.json").read_text())
         self.assertEqual(manifest["split_pairs"], expected)
         (directory / "train.en").write_text("tampered\n", encoding="utf-8")
         with self.assertRaisesRegex(ValueError, "checksum"):
             download_multi30k(folder / "output", base_url=fixture.as_uri(), expected_splits=expected)
 
     def test_failed_download_cleans_partial_file(self):
         folder = self.temporary_directory()
         with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
             with self.assertRaises(urllib.error.URLError):
                 download_multi30k(folder)
         self.assertEqual(list(folder.rglob("*.part")), [])
 
     def test_smoothed_loss_matches_hand_calculation_excluding_pad(self):
         probabilities = torch.tensor([[[.1, .2, .3, .4], [.4, .3, .2, .1]]], dtype=torch.double)
         logits = probabilities.log().requires_grad_()
         loss, nll, count = token_loss(logits, torch.tensor([[3, PAD_ID]]), .1)
         expected = .9 * -math.log(.4) + .1 / 4 * -probabilities[0, 0].log().sum().item()
         self.assertAlmostEqual(loss.item(), expected)
         self.assertAlmostEqual(nll.item(), -math.log(.4))
         self.assertEqual(count, 1)
         loss.backward()
         self.assertEqual(logits.grad[0, 1].abs().sum().item(), 0.)
 
     def test_perplexity_uses_unsmoothed_nll(self):
         p = self.processed()
         class Uniform(nn.Module):
             def forward(self, src, lengths, tgt):
                 return torch.zeros(*tgt.shape, len(p.vocab))
         result = evaluate_perplexity(Uniform(), p, "cpu", epsilon=.1)
         self.assertAlmostEqual(result["perplexity"], len(p.vocab), places=4)
         self.assertEqual(result["tokens"], sum(len(tgt) + 1 for _, tgt in p.dataset.rows))
 
     def test_schedule_boundaries(self):
         c = TransformerConfig()
         self.assertEqual(learning_rate(0, c), 1e-7)
         self.assertAlmostEqual(learning_rate(2000, c), .005)
         self.assertAlmostEqual(learning_rate(8000, c), .0025)
         self.assertEqual(learning_rate(9000, TransformerConfig.smoke()), .001)
 
     def test_unequal_microbatches_equal_combined_token_update(self):
         p = self.processed()
         torch.manual_seed(10)
         split_model = DummyTransformer(len(p.vocab)).double()
         combined_model = copy.deepcopy(split_model)
         batches = [pad_collate([p.dataset[0]]), pad_collate([p.dataset[i] for i in (1, 2, 3)])]
         config = replace(TransformerConfig.smoke(), label_smoothing=.1, grad_clip=1000.)
         split_opt = torch.optim.SGD(split_model.parameters(), lr=.1)
         combined_opt = torch.optim.SGD(combined_model.parameters(), lr=.1)
         train_update(split_model, batches, split_opt, config, "cpu")
         train_update(combined_model, [pad_collate([p.dataset[i] for i in range(4)])], combined_opt, config, "cpu")
         for a, b in zip(split_model.parameters(), combined_model.parameters()):
             torch.testing.assert_close(a, b)
 
     def test_checkpoint_roundtrip_and_random_state(self):
         folder = self.temporary_directory()
         model = DummyTransformer(8)
         opt = torch.optim.Adam(model.parameters())
         torch.manual_seed(11)
         save_checkpoint(folder / "one.pt", model, opt, {"updates": 3}, {"config": {}})
         expected = torch.rand(5)
         with torch.no_grad():
             model.output.weight.zero_()
         loaded = load_checkpoint(folder / "one.pt", model=model, optimizer=opt, restore_rng=True)
         self.assertEqual(loaded["state"]["updates"], 3)
         torch.testing.assert_close(torch.rand(5), expected)
         self.assertGreater(model.output.weight.abs().sum().item(), 0.)
 
     def test_resume_with_dropout_matches_uninterrupted_training(self):
         p, folder = self.processed(), self.temporary_directory()
         class DropoutDummy(DummyTransformer):
             def forward(self, src, lengths, tgt):
                 return self.output(torch.nn.functional.dropout(self.embedding(tgt), .2, self.training))
         torch.manual_seed(123)
         initial = DropoutDummy(len(p.vocab))
         whole, resumed = copy.deepcopy(initial), copy.deepcopy(initial)
         config = replace(TransformerConfig.smoke(), max_epochs=2)
         torch.manual_seed(45)
         full_state = fit(whole, p, p, config, "cpu", output_dir=folder / "whole", verbose=False)
         torch.manual_seed(45)
         fit(resumed, p, p, replace(config, max_epochs=1), "cpu", output_dir=folder / "split", verbose=False)
         state = fit(resumed, p, p, config, "cpu", output_dir=folder / "split",
                     resume=folder / "split/checkpoint_epoch_0001.pt", verbose=False)
         self.assertEqual(state["updates"], full_state["updates"])
         for a, b in zip(whole.parameters(), resumed.parameters()):
             torch.testing.assert_close(a, b, rtol=0, atol=0)
         saved = load_checkpoint(folder / "split/checkpoint_epoch_0002.pt")
         self.assertEqual(saved["metadata"]["config"]["max_epochs"], 2)
 
     def test_mid_epoch_resume_matches_uninterrupted(self):
         p, folder = self.processed(), self.temporary_directory()
         torch.manual_seed(123)
         initial = DummyTransformer(len(p.vocab))
         whole, resumed = copy.deepcopy(initial), copy.deepcopy(initial)
         config = TransformerConfig.smoke()
         fit(whole, p, p, config, "cpu", output_dir=folder / "whole", verbose=False)
         fit(resumed, p, p, replace(config, max_updates=1), "cpu", output_dir=folder / "split", verbose=False)
         state = fit(resumed, p, p, config, "cpu", output_dir=folder / "split",
                     resume=folder / "split/checkpoint_partial.pt", verbose=False)
         self.assertEqual(state["updates"], 2)
         for a, b in zip(whole.parameters(), resumed.parameters()):
             torch.testing.assert_close(a, b, rtol=0, atol=0)
 
     def test_checkpoint_average_and_insufficient_history(self):
         folder = self.temporary_directory()
         model = nn.Linear(1, 1, bias=False)
         opt = torch.optim.Adam(model.parameters())
         paths = []
         for i in range(10):
             with torch.no_grad():
                 model.weight.fill_(i)
             path = folder / f"{i}.pt"
             save_checkpoint(path, model, opt, {"updates": i}, {"vocab": ["same"]})
             paths.append(path)
         output = average_checkpoints(paths, folder / "average.pt")
         self.assertEqual(load_checkpoint(output)["model"]["weight"].item(), 4.5)
         with self.assertRaisesRegex(ValueError, "at least 10"):
             average_checkpoints(paths[:9], folder / "bad.pt")
 
     def test_average_rejects_different_vocabularies(self):
         folder = self.temporary_directory()
         model = nn.Linear(1, 1)
         opt = torch.optim.Adam(model.parameters())
         paths = [folder / "one.pt", folder / "two.pt"]
         for path, vocab in zip(paths, (["one"], ["two"])):
             save_checkpoint(path, model, opt, {}, {"vocab": vocab})
         with self.assertRaisesRegex(ValueError, "different metadata"):
             average_checkpoints(paths, folder / "average.pt", count=2)
 
     def test_beam_search_finds_better_complete_sequence_than_greedy(self):
         class Scripted(nn.Module):
             def forward(self, src, lengths, tgt):
                 logits = torch.full((*tgt.shape, 7), -100.)
                 if tgt.size(1) == 1:
                     logits[:, -1, 4] = math.log(.6)
                     logits[:, -1, 5] = math.log(.4)
                 else:
                     for i in range(tgt.size(0)):
                         probability = .1 if tgt[i, -1] == 4 else .9
                         logits[i, -1, EOS_ID] = math.log(probability)
                         logits[i, -1, 6] = math.log(1 - probability)
                 return logits
         src, lengths = torch.tensor([[4, EOS_ID]]), torch.tensor([2])
         self.assertEqual(generate(Scripted(), src, lengths, beam_size=1, max_tokens=1).tolist(), [[4, EOS_ID]])
         self.assertEqual(generate(Scripted(), src, lengths, beam_size=2, max_tokens=1).tolist(), [[5, EOS_ID]])
 
     def test_eos_minimum_and_batch_output_padding(self):
         class Stop(nn.Module):
             def forward(self, src, lengths, tgt):
                 logits = torch.full((*tgt.shape, 6), -100.)
                 logits[:, -1, EOS_ID] = 10.
                 logits[:, -1, 4] = 9.
                 return logits
         result = generate(Stop(), torch.tensor([[4, EOS_ID], [5, EOS_ID]]), torch.tensor([2, 2]), beam_size=1)
         self.assertEqual(result.tolist(), [[4, EOS_ID], [4, EOS_ID]])
 
     def test_bleu_case_bpe_unknown_and_brevity(self):
         sentence = "a small red house .".split()
         self.assertAlmostEqual(corpus_bleu([sentence], [sentence]), 100.)
         self.assertEqual(corpus_bleu([[]], [sentence]), 0.)
         self.assertEqual(corpus_bleu([["A", "SMALL", "RED", "HOUSE"]], [sentence]), 0.)
         self.assertEqual(corpus_bleu([["<unk>"] * 4], [["<unk>"] * 4]), 0.)
         self.assertEqual(remove_bpe(["ein", "kle@@", "ines", "haus"]), ["ein", "kleines", "haus"])
         self.assertAlmostEqual(corpus_bleu([sentence[:4]], [sentence]), 100 * math.exp(1 - 5 / 4))
 
-    def test_all_model_constructors_are_stubs_and_scripts_discovered(self):
-        modules = {
-            "embeddings": [("TokenEmbedding", (8, 16))],
-            "positional_encoding": [("SinusoidalPositionalEncoding", (16,))],
-            "attention": [("MultiHeadAttention", (16, 2))],
-            "feed_forward": [("PositionwiseFeedForward", (16, 32))],
-            "normalization": [("LayerNorm", (16,)), ("ResidualConnection", (16,))],
-            "encoder": [("EncoderLayer", (TransformerConfig(),)), ("Encoder", (TransformerConfig(),))],
-            "decoder": [("DecoderLayer", (TransformerConfig(),)), ("Decoder", (TransformerConfig(),))],
-            "nmt": [("TransformerNMT", (8, TransformerConfig()))],
-        }
-        for module, constructors in modules.items():
-            for name, args in constructors:
-                with self.subTest(name=name), self.assertRaises(NotImplementedError):
-                    getattr(importlib.import_module(f"transformer.model.{module}"), name)(*args)
         with self.assertRaises(NotImplementedError):
-            scaled_dot_product_attention(None, None, None)
         for stage in ("prepare", "train", "eval"):
             self.assertIn("transformer", available_models(stage))
 
 
 if __name__ == "__main__":
     unittest.main()
```

### After

```diff
+"""Transformer infrastructure and integrated-model checks."""
 import copy
 from dataclasses import replace
 import importlib
 import json
 import math
 from pathlib import Path
 import tempfile
 import unittest
 from unittest.mock import patch
 import urllib.error
 import torch
 from torch import nn
 
 from scripts.dispatch import available_models
 from scripts.paths import REPO_ROOT
 from transformer.config import TransformerConfig
 from transformer.data.corpus import load_pairs, smoke_pairs
 from transformer.data.dataset import process_pairs, pad_collate, TokenBatchSampler
 from transformer.data.download import FILES, download_multi30k
 from transformer.data.masks import causal_mask, padding_mask, decoder_mask
 from transformer.data.vocab import build_vocab, PAD_ID, EOS_ID, UNK_ID
 from transformer.eval.bleu import corpus_bleu, remove_bpe
 from transformer.eval.decoding import generate
 from transformer.eval.metrics import evaluate_perplexity
 from transformer.model.attention import scaled_dot_product_attention
 from transformer.training.checkpoints import save_checkpoint, load_checkpoint, average_checkpoints
 from transformer.training.loop import fit, train_update
 from transformer.training.loss import token_loss
 from transformer.training.schedule import learning_rate
 from transformer.training.smoke import DummyTransformer
 
 
 class TransformerScaffoldingTests(unittest.TestCase):
     @classmethod
     def setUpClass(cls):
         torch.set_num_threads(1)
         cls.scratch = REPO_ROOT / ".tmp/transformer-scaffold-tests"
         cls.scratch.mkdir(parents=True, exist_ok=True)
 
     def temporary_directory(self):
         context = tempfile.TemporaryDirectory(dir=self.scratch)
         self.addCleanup(context.cleanup)
         return Path(context.name)
 
     def processed(self):
         pairs = smoke_pairs()
         return process_pairs(pairs, build_vocab(pairs))
 
     def test_configuration_validation_and_smoke_defaults(self):
         TransformerConfig().validate()
         TransformerConfig.smoke().validate()
         for overrides in ({"heads": 3}, {"max_tokens": 0}, {"seed": -1},
                           {"label_smoothing": 1.}, {"adam_eps": float("nan")},
                           {"warmup_init_lr": .1}):
             with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                 replace(TransformerConfig(), **overrides).validate()
 
     def test_joined_vocab_lexical_ties_and_no_eval_leakage(self):
         vocab = build_vocab([(["z", "a", "a"], ["b", "z"])])
         self.assertEqual(vocab.tokens[:8], ["<s>", "<pad>", "</s>", "<unk>", "a", "z", "b", "madeupword0000"])
         self.assertEqual(vocab.encode(["validation_only"]), [UNK_ID])
         self.assertEqual(len(vocab) % 8, 0)
 
     def test_source_eos_left_padding_target_shift_and_masking(self):
         pairs = [(["a"], ["x", "y"]), (["a", "b", "c"], ["x"])]
         processed = process_pairs(pairs, build_vocab(pairs))
         batch = pad_collate([processed.dataset[i] for i in (0, 1)])
         self.assertEqual(batch["src_lengths"].tolist(), [2, 4])
         self.assertEqual(batch["src"][0, :2].tolist(), [PAD_ID, PAD_ID])
         self.assertEqual(batch["src"][:, -1].tolist(), [EOS_ID, EOS_ID])
         self.assertEqual(batch["tgt_in"][:, 0].tolist(), [EOS_ID, EOS_ID])
         self.assertEqual(batch["tgt_out"][1, 1:].tolist(), [EOS_ID, PAD_ID])
         mask = decoder_mask(batch["tgt_in"])
         self.assertTrue(mask[0, 0, 0, 1])
         self.assertFalse(mask[0, 0, 1, 1])
         self.assertTrue(mask[1, 0, 2, 2])
         self.assertEqual(padding_mask(batch["src"])[0, 0, 0].tolist(), [True, True, False, False])
 
     def test_empty_and_overlong_rows_rejected_without_truncation(self):
         vocab = build_vocab(smoke_pairs())
         for pairs in ([([], ["x"])], [(["a"], [])], []):
             with self.assertRaises(ValueError):
                 process_pairs(pairs, vocab)
         with self.assertRaises(ValueError):
             process_pairs([(["a"] * 5, ["b"])], vocab, max_positions=5)
         self.assertTrue(decoder_mask(torch.full((1, 3), PAD_ID)).all())
         self.assertEqual(causal_mask(3)[0, 0].sum().item(), 3)
 
     def test_parallel_alignment_and_empty_content(self):
         folder = self.temporary_directory()
         src, tgt = folder / "src", folder / "tgt"
         src.write_text("one\ntwo\n", encoding="utf-8")
         tgt.write_text("eins\n", encoding="utf-8")
         with self.assertRaisesRegex(ValueError, "mismatch"):
             load_pairs(src, tgt)
         tgt.write_text("eins\n\n", encoding="utf-8")
         with self.assertRaisesRegex(ValueError, "empty"):
             load_pairs(src, tgt)
 
     def test_token_budget_covers_all_rows_and_preserves_eval_order(self):
         p = self.processed()
         sampler = TokenBatchSampler(p.dataset, 15, seed=11)
         batches = list(sampler)
         self.assertEqual(sorted(i for b in batches for i in b), list(range(4)))
         self.assertEqual(batches, list(sampler))
         for indices in batches:
             batch = pad_collate([p.dataset[i] for i in indices])
             self.assertLessEqual(max(batch["src"].size(1), batch["tgt_in"].size(1)) * len(indices), 15)
+        self.assertEqual(sorted(i for b in TokenBatchSampler(p.dataset, 15, shuffle=False) for i in b), list(range(4)))
         with self.assertRaises(ValueError):
             list(TokenBatchSampler(p.dataset, 1))
 
     def test_cached_download_checksums_and_counts(self):
         folder = self.temporary_directory()
         fixture = folder / "fixture"
         fixture.mkdir()
         for name in FILES:
             (fixture / name).write_text("#version: 0.2\na b\n" if name == "code" else "one two\n", encoding="utf-8")
         expected = {"train": 1, "valid": 1, "test.2016": 1}
         directory = download_multi30k(folder / "output", base_url=fixture.as_uri(), expected_splits=expected)
         with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
             download_multi30k(folder / "output", base_url=fixture.as_uri(), expected_splits=expected)
         manifest = json.loads((directory / "manifest.json").read_text())
         self.assertEqual(manifest["split_pairs"], expected)
         (directory / "train.en").write_text("tampered\n", encoding="utf-8")
         with self.assertRaisesRegex(ValueError, "checksum"):
             download_multi30k(folder / "output", base_url=fixture.as_uri(), expected_splits=expected)
 
     def test_failed_download_cleans_partial_file(self):
         folder = self.temporary_directory()
         with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
             with self.assertRaises(urllib.error.URLError):
                 download_multi30k(folder)
         self.assertEqual(list(folder.rglob("*.part")), [])
 
     def test_smoothed_loss_matches_hand_calculation_excluding_pad(self):
         probabilities = torch.tensor([[[.1, .2, .3, .4], [.4, .3, .2, .1]]], dtype=torch.double)
         logits = probabilities.log().requires_grad_()
         loss, nll, count = token_loss(logits, torch.tensor([[3, PAD_ID]]), .1)
         expected = .9 * -math.log(.4) + .1 / 4 * -probabilities[0, 0].log().sum().item()
         self.assertAlmostEqual(loss.item(), expected)
         self.assertAlmostEqual(nll.item(), -math.log(.4))
         self.assertEqual(count, 1)
         loss.backward()
         self.assertEqual(logits.grad[0, 1].abs().sum().item(), 0.)
 
     def test_perplexity_uses_unsmoothed_nll(self):
         p = self.processed()
         class Uniform(nn.Module):
             def forward(self, src, lengths, tgt):
                 return torch.zeros(*tgt.shape, len(p.vocab))
         result = evaluate_perplexity(Uniform(), p, "cpu", epsilon=.1)
         self.assertAlmostEqual(result["perplexity"], len(p.vocab), places=4)
         self.assertEqual(result["tokens"], sum(len(tgt) + 1 for _, tgt in p.dataset.rows))
 
     def test_schedule_boundaries(self):
         c = TransformerConfig()
         self.assertEqual(learning_rate(0, c), 1e-7)
         self.assertAlmostEqual(learning_rate(2000, c), .005)
         self.assertAlmostEqual(learning_rate(8000, c), .0025)
         self.assertEqual(learning_rate(9000, TransformerConfig.smoke()), .001)
 
     def test_unequal_microbatches_equal_combined_token_update(self):
         p = self.processed()
         torch.manual_seed(10)
         split_model = DummyTransformer(len(p.vocab)).double()
         combined_model = copy.deepcopy(split_model)
         batches = [pad_collate([p.dataset[0]]), pad_collate([p.dataset[i] for i in (1, 2, 3)])]
         config = replace(TransformerConfig.smoke(), label_smoothing=.1, grad_clip=1000.)
         split_opt = torch.optim.SGD(split_model.parameters(), lr=.1)
         combined_opt = torch.optim.SGD(combined_model.parameters(), lr=.1)
         train_update(split_model, batches, split_opt, config, "cpu")
         train_update(combined_model, [pad_collate([p.dataset[i] for i in range(4)])], combined_opt, config, "cpu")
         for a, b in zip(split_model.parameters(), combined_model.parameters()):
             torch.testing.assert_close(a, b)
 
     def test_checkpoint_roundtrip_and_random_state(self):
         folder = self.temporary_directory()
         model = DummyTransformer(8)
         opt = torch.optim.Adam(model.parameters())
         torch.manual_seed(11)
         save_checkpoint(folder / "one.pt", model, opt, {"updates": 3}, {"config": {}})
         expected = torch.rand(5)
         with torch.no_grad():
             model.output.weight.zero_()
         loaded = load_checkpoint(folder / "one.pt", model=model, optimizer=opt, restore_rng=True)
         self.assertEqual(loaded["state"]["updates"], 3)
         torch.testing.assert_close(torch.rand(5), expected)
         self.assertGreater(model.output.weight.abs().sum().item(), 0.)
 
     def test_resume_with_dropout_matches_uninterrupted_training(self):
         p, folder = self.processed(), self.temporary_directory()
         class DropoutDummy(DummyTransformer):
             def forward(self, src, lengths, tgt):
                 return self.output(torch.nn.functional.dropout(self.embedding(tgt), .2, self.training))
         torch.manual_seed(123)
         initial = DropoutDummy(len(p.vocab))
         whole, resumed = copy.deepcopy(initial), copy.deepcopy(initial)
         config = replace(TransformerConfig.smoke(), max_epochs=2)
         torch.manual_seed(45)
         full_state = fit(whole, p, p, config, "cpu", output_dir=folder / "whole", verbose=False)
         torch.manual_seed(45)
         fit(resumed, p, p, replace(config, max_epochs=1), "cpu", output_dir=folder / "split", verbose=False)
         state = fit(resumed, p, p, config, "cpu", output_dir=folder / "split",
                     resume=folder / "split/checkpoint_epoch_0001.pt", verbose=False)
         self.assertEqual(state["updates"], full_state["updates"])
         for a, b in zip(whole.parameters(), resumed.parameters()):
             torch.testing.assert_close(a, b, rtol=0, atol=0)
         saved = load_checkpoint(folder / "split/checkpoint_epoch_0002.pt")
         self.assertEqual(saved["metadata"]["config"]["max_epochs"], 2)
 
     def test_mid_epoch_resume_matches_uninterrupted(self):
         p, folder = self.processed(), self.temporary_directory()
         torch.manual_seed(123)
         initial = DummyTransformer(len(p.vocab))
         whole, resumed = copy.deepcopy(initial), copy.deepcopy(initial)
         config = TransformerConfig.smoke()
         fit(whole, p, p, config, "cpu", output_dir=folder / "whole", verbose=False)
         fit(resumed, p, p, replace(config, max_updates=1), "cpu", output_dir=folder / "split", verbose=False)
         state = fit(resumed, p, p, config, "cpu", output_dir=folder / "split",
                     resume=folder / "split/checkpoint_partial.pt", verbose=False)
         self.assertEqual(state["updates"], 2)
         for a, b in zip(whole.parameters(), resumed.parameters()):
             torch.testing.assert_close(a, b, rtol=0, atol=0)
 
     def test_checkpoint_average_and_insufficient_history(self):
         folder = self.temporary_directory()
         model = nn.Linear(1, 1, bias=False)
         opt = torch.optim.Adam(model.parameters())
         paths = []
         for i in range(10):
             with torch.no_grad():
                 model.weight.fill_(i)
             path = folder / f"{i}.pt"
             save_checkpoint(path, model, opt, {"updates": i}, {"vocab": ["same"]})
             paths.append(path)
         output = average_checkpoints(paths, folder / "average.pt")
         self.assertEqual(load_checkpoint(output)["model"]["weight"].item(), 4.5)
         with self.assertRaisesRegex(ValueError, "at least 10"):
             average_checkpoints(paths[:9], folder / "bad.pt")
 
     def test_average_rejects_different_vocabularies(self):
         folder = self.temporary_directory()
         model = nn.Linear(1, 1)
         opt = torch.optim.Adam(model.parameters())
         paths = [folder / "one.pt", folder / "two.pt"]
         for path, vocab in zip(paths, (["one"], ["two"])):
             save_checkpoint(path, model, opt, {}, {"vocab": vocab})
         with self.assertRaisesRegex(ValueError, "different metadata"):
             average_checkpoints(paths, folder / "average.pt", count=2)
 
     def test_beam_search_finds_better_complete_sequence_than_greedy(self):
         class Scripted(nn.Module):
             def forward(self, src, lengths, tgt):
                 logits = torch.full((*tgt.shape, 7), -100.)
                 if tgt.size(1) == 1:
                     logits[:, -1, 4] = math.log(.6)
                     logits[:, -1, 5] = math.log(.4)
                 else:
                     for i in range(tgt.size(0)):
                         probability = .1 if tgt[i, -1] == 4 else .9
                         logits[i, -1, EOS_ID] = math.log(probability)
                         logits[i, -1, 6] = math.log(1 - probability)
                 return logits
         src, lengths = torch.tensor([[4, EOS_ID]]), torch.tensor([2])
         self.assertEqual(generate(Scripted(), src, lengths, beam_size=1, max_tokens=1).tolist(), [[4, EOS_ID]])
         self.assertEqual(generate(Scripted(), src, lengths, beam_size=2, max_tokens=1).tolist(), [[5, EOS_ID]])
 
     def test_eos_minimum_and_batch_output_padding(self):
         class Stop(nn.Module):
             def forward(self, src, lengths, tgt):
                 logits = torch.full((*tgt.shape, 6), -100.)
                 logits[:, -1, EOS_ID] = 10.
                 logits[:, -1, 4] = 9.
                 return logits
         result = generate(Stop(), torch.tensor([[4, EOS_ID], [5, EOS_ID]]), torch.tensor([2, 2]), beam_size=1)
         self.assertEqual(result.tolist(), [[4, EOS_ID], [4, EOS_ID]])
 
     def test_bleu_case_bpe_unknown_and_brevity(self):
         sentence = "a small red house .".split()
         self.assertAlmostEqual(corpus_bleu([sentence], [sentence]), 100.)
         self.assertEqual(corpus_bleu([[]], [sentence]), 0.)
         self.assertEqual(corpus_bleu([["A", "SMALL", "RED", "HOUSE"]], [sentence]), 0.)
         self.assertEqual(corpus_bleu([["<unk>"] * 4], [["<unk>"] * 4]), 0.)
         self.assertEqual(remove_bpe(["ein", "kle@@", "ines", "haus"]), ["ein", "kleines", "haus"])
         self.assertAlmostEqual(corpus_bleu([sentence[:4]], [sentence]), 100 * math.exp(1 - 5 / 4))
 
+    def test_real_model_forward_backward_and_scripts_discovered(self):
+        from transformer.model.nmt import TransformerNMT
+        from transformer.model.normalization import ResidualConnection
+        c = replace(TransformerConfig.smoke(), encoder_layers=2, decoder_layers=2)
+        model = TransformerNMT(16, c)
+        src = torch.tensor([[PAD_ID, 4, EOS_ID], [5, 6, EOS_ID]])
+        tgt = torch.tensor([[EOS_ID, 7], [EOS_ID, 8]])
+        logits = model(src, src.ne(PAD_ID).sum(1), tgt)
+        self.assertEqual(logits.shape, (2, 2, 16))
+        self.assertTrue(torch.isfinite(logits).all())
+        logits.square().sum().backward()
+        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
+        self.assertIs(model.output_projection.weight, model.token_embeddings.embedding.weight)
         with self.assertRaises(NotImplementedError):
+            ResidualConnection(16)
         for stage in ("prepare", "train", "eval"):
             self.assertIn("transformer", available_models(stage))
 
 
 if __name__ == "__main__":
     unittest.main()
```

## Validation: reference convention regression tests

File: `tests/test_transformer_reference.py`

New regression tests cover positional frequencies, padding offsets and odd model widths. They also verify the reference Adam update against a hand calculation and check attention biases and fully blocked rows. These tests target numerical conventions that can silently diverge even when tensor shapes and a basic forward pass look correct.

### Before

New file: there was no before code.

### After

```diff
+"""Regression checks for the pinned Multi30k reference conventions."""
+import math
+import unittest
+import torch
+from transformer.model.positional_encoding import SinusoidalPositionalEncoding
+from transformer.training.adam import ReferenceAdam
+from transformer.model.attention import MultiHeadAttention
+
+
+class ReferenceTests(unittest.TestCase):
+    def test_positions_frequency_offset_padding_and_odd_dimension(self):
+        for d in (7, 8):
+            m = SinusoidalPositionalEncoding(d, 4, 0.)
+            mask = torch.tensor([[True, True, False, False], [False, False, True, True]])
+            result = m(torch.zeros(2, 4, d), mask)
+            torch.testing.assert_close(result[0, 2:], result[1, :2])
+            self.assertEqual(result[0, :2].count_nonzero(), 0)
+            self.assertAlmostEqual(float(result[0, 2, 0]), math.sin(2), places=6)
+            self.assertAlmostEqual(float(result[0, 2, d//2]), math.cos(2), places=6)
+            self.assertAlmostEqual(float(result[0, 2, d//2-1]), math.sin(2/10000), places=6)
+            self.assertTrue(torch.isfinite(m(torch.zeros(1, 4, d), torch.zeros(1, 4, dtype=torch.bool))).all())
+            if d % 2:
+                self.assertEqual(result[..., -1].count_nonzero(), 0)
+
+    def test_adam_epsilon_before_bias_correction(self):
+        p = torch.nn.Parameter(torch.tensor([.2], dtype=torch.double))
+        opt = ReferenceAdam([p], lr=.1, betas=(.9, .98), eps=.01)
+        p.grad = torch.tensor([.3], dtype=torch.double)
+        opt.step()
+        expected = .2 - (.1 * math.sqrt(.02) / .1) * (.1 * .3) / (math.sqrt(.02 * .3**2) + .01)
+        self.assertAlmostEqual(p.item(), expected, places=14)
+
+    def test_attention_biases_and_blocked_rows(self):
+        m = MultiHeadAttention(8, 2)
+        self.assertIsNotNone(m.q_proj.bias)
+        self.assertEqual(m.o_proj.bias.count_nonzero(), 0)
+        x = torch.randn(2, 3, 8, requires_grad=True)
+        out = m(x, x, x, torch.ones(2, 1, 3, 3, dtype=torch.bool))
+        self.assertTrue(torch.isfinite(out.context).all())
+        self.assertEqual(out.weights.count_nonzero(), 0)
+        out.context.sum().backward()
+        self.assertTrue(torch.isfinite(x.grad).all())
```

## Additional documentation

- `scripts/transformer/REFERENCE.md`: reference revision, aligned settings, commands, parity evidence, and remaining runtime differences.
- This report: complete before/after code.

This report records the implementation frozen for the benchmark run; source code was not changed during that run. Later decoder comments and decoding refactoring are separate changes and are not incorporated into these frozen before/after listings.
