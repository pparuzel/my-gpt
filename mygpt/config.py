from dataclasses import dataclass
from pathlib import Path


@dataclass
class GPTConfig:
    ctx_len: int
    emb_dim: int
    num_heads: int
    num_blocks: int
    dropout: float


@dataclass
class TrainingConfig:
    dataset: str
    batch_size: int
    learning_rate: float
    epochs: int
    l2_reg: float
    eval_interval: int
    eval_iters: int
    train_val_split: float
    vocab_size: int
    seed: int
    load_checkpoint: Path | None = None
    save_checkpoint: str | None = None
