import math

import torch


class PrecomputedPositionalEncoding(torch.nn.Module):
    LN_10000 = math.log(10000.0)

    def __init__(self, max_len: int, emb_dim: int) -> None:
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.pe: torch.Tensor
        self.register_buffer("pe", self._precompute())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]

    def _precompute(self) -> torch.Tensor:
        """Positional encodings as in the 'Attention Is All You Need' paper.

        Returns: tensor of shape [max_len, emb_dim].

        """
        pe = torch.zeros(self.max_len, self.emb_dim)
        position = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(1)
        dims = torch.arange(0, self.emb_dim, 2, dtype=torch.float32)
        div_term = torch.exp(dims * (-self.LN_10000 / self.emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)


def add_positional_encoding_layer(
    max_len: int,
    emb_dim: int,
    *,
    learnable: bool = False,
) -> torch.nn.Module:
    """Add positional encoding layer.

    Args:
        max_len: Maximum length of the sequence.
        emb_dim: Embedding dimension (Channel).

    Returns:
        Layer that adds positional encoding to the input tensor.

    """
    if learnable:
        # A position-learnable embedding space.
        return torch.nn.Embedding(max_len, emb_dim)

    return PrecomputedPositionalEncoding(max_len, emb_dim)
