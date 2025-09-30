import torch

from mygpt.attention import CausalSelfAttention


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        *,
        emb_dim: int,
        ctx_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if emb_dim % num_heads != 0:
            msg = (
                f"Embedding dimension ({emb_dim}) must be "
                f"divisible by num_heads ({num_heads})"
            )
            raise ValueError(msg)
        self.attn = CausalSelfAttention(
            num_heads=num_heads,
            emb_dim=emb_dim,
            ctx_len=ctx_len,
            dropout=dropout,
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim * 4),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(emb_dim * 4, emb_dim),
            torch.nn.Dropout(dropout),
        )
        self.ln1 = torch.nn.LayerNorm(emb_dim)
        self.ln2 = torch.nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unlike the original Transformer (Post-LayerNorm: LN after residual),
        # GPT-family use Pre-LayerNorm (LN before attention/FFN) because it
        # stabilizes training and gradient flow in deep networks, making
        # convergence more reliable at scale.

        # Multiple Self-attention heads. (Batch, Time, head_size).
        x = x + self.attn(self.ln1(x))
        # Feed-forward layer to reason about attention outputs.
        return x + self.mlp(self.ln2(x))
