import torch


class CausalSelfAttention(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        *,
        emb_dim: int,
        ctx_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        """Instead of creating 3 separate linear layers for Q, K, and V we can
        use a single linear layer that outputs all three in one go:
        ```
        self.query = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.key = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        ```
        """
        self.qkv_proj = torch.nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.out_proj = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.num_heads = num_heads
        self.tril: torch.Tensor
        tril = torch.tril(torch.ones(1, 1, ctx_len, ctx_len))
        self.register_buffer("tril", tril)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        head_size = c // self.num_heads
        # Query, Key, and Value projections. (Batch, Time, head_size).
        # Head size is an even chunk of emb_dim per each head.
        query, key, value = self.qkv_proj(x).chunk(3, dim=-1)
        # Multi-head attention. (Batch, Head, Time, head_size).
        query = query.view(b, t, self.num_heads, head_size).transpose(1, 2)
        key = key.view(b, t, self.num_heads, head_size).transpose(1, 2)
        value = value.view(b, t, self.num_heads, head_size).transpose(1, 2)
        # Token affinities. (Batch, Head, Time, Time).
        affinities = query @ key.transpose(-1, -2)
        # Avoids extremely large dot products with normalization scaling.
        affinities *= c**0.5
        # A decoder needs to be autoregressive, hence causal masking.
        # Softmax zeroes out the weights for illegal connections (-inf).
        attention = affinities.masked_fill(
            self.tril[:, :, :t, :t] == 0,
            float("-inf"),
        )
        # Attention matrix for each query and key (Batch, Head, Time, Time).
        attention = torch.softmax(attention, dim=-1)
        # Dropout for regularization.
        attention = self.dropout(attention)
        # Attention output (Batch, Head, Time, head_size).
        output = attention @ value
        # Re-assemble all head outputs side by side (B, H, T, hs).
        output = output.transpose(1, 2).contiguous()  # (B, T, H, hs)
        output = output.view(b, t, c)  # (B, T, C)
        output = self.out_proj(output)
        return self.dropout(output)
