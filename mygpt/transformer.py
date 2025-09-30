import torch

from mygpt.attention_blocks import AttentionBlock
from mygpt.config import GPTConfig, TrainingConfig
from mygpt.positional_encodings import add_positional_encoding_layer


class Transformer(torch.nn.Module):
    def __init__(
        self,
        cfg: GPTConfig,
        train_cfg: TrainingConfig,
        *,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.ctx_len = cfg.ctx_len  # Karpathy's block_size
        self.vocab_size = vocab_size
        # Architecture.
        self.token_embeddings = torch.nn.Embedding(vocab_size, cfg.emb_dim)
        # The learnable positional embeddings vs precomputed positional encoding
        # from the "Attention Is All You Need" paper.
        self.posit_embeddings = add_positional_encoding_layer(
            cfg.ctx_len,
            cfg.emb_dim,
            learnable=False,
        )
        blocks = [
            AttentionBlock(
                num_heads=cfg.num_heads,
                emb_dim=cfg.emb_dim,
                ctx_len=cfg.ctx_len,
                dropout=train_cfg.dropout,
            )
            for _ in range(cfg.num_blocks)
        ]
        self.blocks = torch.nn.Sequential(*blocks)
        self.block_norm = torch.nn.LayerNorm(cfg.emb_dim)
        # Output projection onto vocabulary space.
        self.vocab_proj = torch.nn.Linear(cfg.emb_dim, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _b, t = idx.shape
        device = idx.device

        # (Batch, Time, Channel) where channels are feature dimensions.
        token_embeddings = self.token_embeddings(idx)
        batched_positions = torch.arange(0, t, device=device).unsqueeze(0)
        # (Time, Channel).
        posit_embeddings = self.posit_embeddings(batched_positions)
        # Position-enriched embeddings. (Batch, Time, Channel).
        x = token_embeddings + posit_embeddings
        # Residual blocks.
        x = self.blocks(x)
        # Normalization before the output layer.
        x = self.block_norm(x)
        # Output logits (Batch, Time, vocab_size).
        return self.vocab_proj(x)
