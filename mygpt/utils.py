import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import torch

import mygpt
from mygpt.config import GPTConfig, TrainingConfig

DATA_DIR = Path(mygpt.__file__).parent.parent / "data"
MODELS_DIR = Path(mygpt.__file__).parent.parent / "checkpoints"


class SplitData(NamedTuple):
    train: torch.Tensor
    validate: torch.Tensor


def split_into_train_val(
    data: torch.Tensor,
    train_perc: float,
) -> SplitData:
    n = int(len(data) * train_perc)  # Split point
    return SplitData(train=data[:n], validate=data[n:])


def read_dataset(name: str | Path) -> str:
    if isinstance(name, str):
        name = DATA_DIR / name
    return name.read_text()


def as_ckpt_path(path: str | None) -> Path | None:
    if path is None:
        return None
    return _as_ckpt_path(path)


def _as_ckpt_path(path: str) -> Path:
    p = MODELS_DIR / path
    if p.suffix != ".pt":
        p = p.parent / (p.name + ".pt")
    if not p.exists():
        msg = f"Checkpoint path {p} does not exist"
        raise FileNotFoundError(msg)
    return p


def save_model(
    model: mygpt.GPT,
    config: GPTConfig,
    train_config: TrainingConfig,
) -> None:
    name = train_config.save_checkpoint
    if not name:
        date_str = datetime.now().strftime(r"%Y.%m.%d_%H%M%S")  # noqa: DTZ005
        name = (
            f"gpt.c{config.ctx_len}"
            f".e{config.emb_dim}"
            f".h{config.num_heads}"
            f".b{config.num_blocks}"
            f".d{config.dropout}"
            f"-{date_str}.pt"
        )
    torch.save(model, MODELS_DIR / name)
    print("=== Saving model ===")
    print(name)


def load_model(name: str | Path, *, weights_only: bool = False) -> mygpt.GPT:
    if isinstance(name, str):
        name = _as_ckpt_path(name)
        name = (MODELS_DIR / name).with_suffix(".pt")

    print("=== Loading model ===")
    model = torch.load(name, weights_only=weights_only, map_location="cpu")
    if not isinstance(model, mygpt.GPT):
        msg = f"Loaded object is not a GPT model, got {type(model)}"
        raise TypeError(msg)
    return model


def handle_backend(backend: str | None) -> str:
    apple = sys.platform == "darwin"
    if backend is None:
        if apple and torch.mps.is_available():
            # M series MacBooks have MPS backend.
            backend = "mps"
        else:
            warnings.warn("No backend specified, using CPU", stacklevel=2)
            backend = "cpu"
    return backend


def handle_randomness(seed: int = -1) -> int:
    seed = seed if seed > 0 else int(torch.randint(0, 2**32 - 1, size=(1,)))
    torch.manual_seed(seed)
    return seed
