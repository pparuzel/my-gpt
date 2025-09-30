import warnings
from typing import overload

import torch

from mygpt.config import GPTConfig, TrainingConfig
from mygpt.tokenizers import Tokenizer
from mygpt.transformer import Transformer

type TorchDevice = str | torch.device | int


class GPT(torch.nn.Module):
    def __init__(
        self,
        config: GPTConfig,
        train_config: TrainingConfig,
        tokenizer: Tokenizer,
        *,
        device: TorchDevice | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.transformer = Transformer(
            config,
            train_config,
            vocab_size=tokenizer.vocab_size,
        ).to(device)

    @overload
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if y is not None:
            y_hat = self.transformer(x)
            loss = torch.nn.functional.cross_entropy(
                y_hat.view(-1, y_hat.size(-1)),
                y.view(-1),
            )
            return y_hat, loss
        return self.transformer(x)

    def generate(
        self,
        idx: torch.Tensor,
        max_tokens: int = 150,
    ) -> torch.Tensor:
        if idx.size(-1) > self.transformer.ctx_len:
            warnings.warn(
                "Context length is insufficient to digest the full input data",
                stacklevel=2,
            )
        for _ in range(max_tokens):
            idx_clipped = idx[:, -self.transformer.ctx_len :]
            logits = self(idx_clipped)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
