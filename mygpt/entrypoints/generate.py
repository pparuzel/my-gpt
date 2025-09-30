import argparse
from pprint import pprint

import torch

from mygpt import GPT
from mygpt.defaults import Defaults
from mygpt.gpt import TorchDevice
from mygpt.utils import (
    as_ckpt_path,
    handle_backend,
    handle_randomness,
    load_model,
)


def generate(model: GPT, max_tokens: int, *, device: TorchDevice) -> None:
    print(f"=== Generation ({max_tokens=}) ===")
    model.eval()
    with torch.inference_mode():
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(context, max_tokens).view(-1).tolist()
        print(model.tokenizer.decode(generated))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        type=as_ckpt_path,
        default=None,
        dest="model_name",
        metavar="model_name",
        help="Load model from checkpoint",
    )
    parser.add_argument(
        "num_tokens",
        type=int,
        default=0,
        nargs="?",
        help=f"Number of tokens to generate [{Defaults.MAX_TOKENS}]",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["mps", "cpu", "cuda"],
        help="Device backend to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        metavar="value",
        help="Seed for random number generator",
    )
    args = parser.parse_args()
    model_name = args.model_name
    seed = handle_randomness(args.seed)
    print(f"Seed: {seed}")

    device = handle_backend(args.backend)
    model = load_model(model_name)
    model.to(device)

    pprint(model.config)
    generate(
        model,
        max_tokens=args.num_tokens or Defaults.MAX_TOKENS,
        device=device,
    )


if __name__ == "__main__":
    main()
