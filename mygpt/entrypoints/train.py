import argparse
from pprint import pprint

import torch
import tqdm

from mygpt import GPT
from mygpt.config import GPTConfig, TrainingConfig
from mygpt.data_processor import DataProcessor
from mygpt.defaults import Defaults
from mygpt.entrypoints.generate import generate
from mygpt.tokenizers import CharTokenizer
from mygpt.utils import (
    as_ckpt_path,
    as_existing_ckpt_path,
    handle_backend,
    handle_randomness,
    load_model,
    save_model,
)


@torch.no_grad()
def generate_loss(
    model: GPT,
    data: DataProcessor,
    batch_size: int,
    *,
    eval_iters: int,
) -> dict[str, float]:
    """More accurate valuation of loss, averaged over `eval_iters` batches."""
    out = {}
    model.eval()  # Disables Dropout, LayerNorm uses running stats
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = data.get_random_batch(split, batch_size=batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()  # Average over eval_iters
    model.train()
    return out


def train(
    model: GPT,
    data: DataProcessor,
    train_config: TrainingConfig,
) -> GPT:
    # Prepare for training.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        # L2 regularization to avoid overfitting
        weight_decay=train_config.l2_reg,
    )
    train_progress = tqdm.trange(0, train_config.epochs)
    batch_size = train_config.batch_size

    # Main loop.
    for epoch in train_progress:
        optimizer.zero_grad()
        x, y = data.get_random_batch("train", batch_size=batch_size)
        _, loss = model(x, y)
        train_progress.set_description(f"Loss {loss.item():.4f}")
        if epoch % train_config.eval_interval == 0:
            losses = generate_loss(
                model,
                data,
                batch_size=batch_size,
                eval_iters=train_config.eval_iters,
            )
            train_progress.set_postfix({"epoch": epoch, **losses})
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate",
        "-g",
        type=int,
        nargs="?",
        default=0,
        metavar="num_tokens",
        help=(
            f"Generate a specified number of tokens after training "
            f"[{Defaults.MAX_TOKENS}]"
        ),
    )
    parser.add_argument(
        "--load",
        type=as_existing_ckpt_path,
        default=None,
        dest="model_name",
        metavar="model_name",
        help="Load model from checkpoint. Default trains a new model",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=as_ckpt_path,
        default=None,
        metavar="model_name",
        help=(
            "Store model with a specified name under checkpoints/ or"
            "an absolute path"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=Defaults.DATASET,
        metavar="filename",
        help=f"Dataset to use from data/ directory [{Defaults.DATASET}]",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Do not save the model after training",
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
        help="Seed for random number generator [random seed]",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        metavar="count",
        help=f"Number of training epochs [{Defaults.EPOCHS}]",
    )
    parser.add_argument(
        "--batch-size",
        "--bs",
        "-B",
        type=int,
        default=0,
        metavar="value",
        help=f"Batch size for training [{Defaults.BATCH_SIZE}]",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        "-l",
        type=float,
        default=0.0,
        metavar="rate",
        help=f"Learning rate. [{Defaults.LEARNING_RATE}]",
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=0.0,
        metavar="rate",
        help=f"L2 regularization factor [{Defaults.L2_REG}]",
    )
    parser.add_argument(
        "--ctx",
        "-c",
        type=int,
        default=0,
        metavar="len",
        help=f"Context length [{Defaults.CTX_LEN}]",
    )
    parser.add_argument(
        "--emb",
        "-e",
        type=int,
        default=0,
        metavar="dims",
        help=f"Embedding dimensions [{Defaults.EMB_DIM}]",
    )
    parser.add_argument(
        "--heads",
        "--hs",
        type=int,
        default=0,
        metavar="count",
        help=f"Number of attention heads [{Defaults.NUM_HEADS}]",
    )
    parser.add_argument(
        "--blocks",
        "--bl",
        "-b",
        type=int,
        default=0,
        metavar="count",
        help=f"Number of transformer blocks [{Defaults.NUM_BLOCKS}]",
    )
    parser.add_argument(
        "--drop",
        "-d",
        type=float,
        default=0.0,
        metavar="rate",
        help=f"Dropout rate [{Defaults.DROPOUT}]",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=0,
        metavar="count",
        help=(
            f"Number of iterations to sample an estimated loss "
            f"[{Defaults.EVAL_ITERS}]"
        ),
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        metavar="interval",
        help=(
            f"Iteration interval to run evaluation of an estimated loss "
            f"[{Defaults.EVAL_INTERVAL}]"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    device = handle_backend(args.backend)
    seed = handle_randomness(args.seed)
    no_save = args.no_save
    load_from = args.model_name
    save_to = args.output
    if save_to and not save_to.name.endswith(".pt"):
        save_to = save_to.parent / (save_to.name + ".pt")
    dataset_name = args.dataset
    train_val_split = Defaults.TRAIN_VAL_SPLIT  # hard-coded for now
    num_tokens = args.generate or Defaults.MAX_TOKENS
    ctx_len = args.ctx or Defaults.CTX_LEN

    data = DataProcessor(
        CharTokenizer,
        dataset_name,
        train_val_split=train_val_split,
        ctx_len=ctx_len,
        device=device,
    )
    train_config = TrainingConfig(
        dataset=dataset_name,
        batch_size=args.batch_size or Defaults.BATCH_SIZE,
        learning_rate=args.learning_rate or Defaults.LEARNING_RATE,
        dropout=args.drop or Defaults.DROPOUT,
        epochs=args.epochs or Defaults.EPOCHS,
        l2_reg=args.l2_reg or Defaults.L2_REG,
        eval_interval=args.eval_interval or Defaults.EVAL_INTERVAL,
        eval_iters=args.eval_iters or Defaults.EVAL_ITERS,
        train_val_split=train_val_split,
        vocab_size=data.tokenizer.vocab_size,
        load_checkpoint=load_from,
        save_checkpoint=save_to,
        seed=seed,
    )

    if train_config.load_checkpoint:
        model = load_model(train_config.load_checkpoint)
        config = model.config  # override config from loaded model
        # We should consider saving the model under the same name.
        train_config.save_checkpoint = train_config.load_checkpoint or save_to
    else:
        config = GPTConfig(
            ctx_len=ctx_len,
            emb_dim=args.emb or Defaults.EMB_DIM,
            num_heads=args.heads or Defaults.NUM_HEADS,
            num_blocks=args.blocks or Defaults.NUM_BLOCKS,
        )
        model = GPT(config, train_config, data.tokenizer, device=device)
    model.to(device)

    print("=== Training ===")
    print()
    pprint(config)
    pprint(train_config)

    try:
        train(model, data, train_config)
    except KeyboardInterrupt:
        print("=== Training interrupted ===")
        print()
        question = "Would you like to save the model? [y]/n: "
        ans = input(question).strip().lower() or "y"
        no_save = ans != "y"
    if not no_save:
        save_model(model, config, train_config)
    if args.generate:
        generate(
            model,
            max_tokens=num_tokens,
            device=device,
        )


if __name__ == "__main__":
    main()
