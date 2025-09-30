import torch

from mygpt.gpt import TorchDevice
from mygpt.tokenizers import Tokenizer
from mygpt.utils import read_dataset, split_into_train_val


class DataProcessor:
    def __init__(
        self,
        tokenizer_cls: type[Tokenizer],
        dataset_name: str,
        *,
        train_val_split: float,
        ctx_len: int,
        device: TorchDevice,
    ) -> None:
        content = read_dataset(dataset_name)
        self.device = device
        self.ctx_len = ctx_len
        self.tokenizer = tokenizer_cls.from_data(content)
        encoded_content = torch.tensor(
            self.tokenizer.encode(content),
            dtype=torch.long,
        )
        train, val = split_into_train_val(
            encoded_content,
            train_perc=train_val_split,
        )
        self._data = {"train": train.to(device), "val": val.to(device)}

    def get_random_batch(
        self,
        data_type: str,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx_len = self.ctx_len
        data = self._data[data_type]
        start_idx = torch.randint(0, len(data) - ctx_len, (batch_size,))
        x = torch.stack([data[i : i + ctx_len] for i in start_idx])
        y = torch.stack([data[i + 1 : i + ctx_len + 1] for i in start_idx])
        return x, y
