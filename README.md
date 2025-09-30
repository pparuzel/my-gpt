# My Toy GPT from Scratch

A complete implementation of GPT (Generative Pre-trained Transformer) in PyTorch, built from the ground up. This project provides a clean, educational implementation of the transformer architecture with training and text generation capabilities.

## Features

- **Complete GPT Implementation**: Full transformer architecture with multi-head attention, positional encodings, and layer normalization
- **Flexible Training**: Configurable model architecture, training parameters, and dataset support
- **Text Generation**: Generate text using trained models with customizable parameters
- **Multiple Datasets**: Includes Shakespeare, Harry Potter, and Avatar script datasets
- **Device Support**: Compatible with CPU, CUDA (NVIDIA GPUs), and MPS (Apple Silicon)
- **Checkpointing**: Save and load model checkpoints for continued training or inference

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Make sure you have `uv` installed:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install dependencies:

```bash
cd my-gpt
uv sync
```

## Usage

### Training a Model

Train a new GPT model from scratch:

```bash
# Basic training with default parameters
uv run training

# Train with custom parameters
uv run training --epochs 1000 --batch-size 64 --learning-rate 0.0005 --ctx 128 --emb 256 --heads 8 --blocks 4

# Train on a specific dataset
uv run training --dataset harry_potter.txt --epochs 2000

# Save the model with a custom name
uv run training --output my_shakespeare_model --epochs 1000
```

#### Training Parameters

- `--epochs`: Number of training epochs (default: 5000)
- `--batch-size, -B`: Batch size for training (default: 128)
- `--learning-rate, --lr, -l`: Learning rate (default: 0.001)
- `--ctx, -c`: Context length (default: 64)
- `--emb, -e`: Embedding dimensions (default: 128)
- `--heads`: Number of attention heads (default: 16)
- `--blocks, --bl, -b`: Number of transformer blocks (default: 1)
- `--drop, -d`: Dropout rate (default: 0.2)
- `--l2-reg`: L2 regularization factor (default: 0.01)
- `--dataset`: Dataset file from `data/` directory (default: tiny_shakespeare.txt)
- `--backend`: Device backend: cpu, cuda, or mps
- `--seed`: Random seed for reproducibility
- `--no-save`: Don't save the model after training
- `--generate, -g`: Generate tokens after training (optional)

### Loading and Continuing Training

Continue training from a saved checkpoint:

```bash
# Load and continue training
uv run training --load gpt.c64.e128.h16.b4-2025.09.30_194304.pt --epochs 1000

# Load a model and generate text immediately after training
uv run training --load my_model --generate 1000
```

### Text Generation

Generate text using a trained model:

```bash
# Generate 500 tokens (default)
uv run generation --load gpt.c64.e128.h16.b4-2025.09.30_194304.pt

# Generate a specific number of tokens
uv run generation 1000 --load my_shakespeare_model

# Generate with a specific seed for reproducibility
uv run generation 500 --load my_model --seed 42

# Specify device backend
uv run generation --load my_model --backend mps
```

#### Generation Parameters

- `num_tokens`: Number of tokens to generate (default: 500)
- `--load`: Model checkpoint to load (required)
- `--backend`: Device backend: cpu, cuda, or mps
- `--seed`: Random seed for reproducible generation

## Available Datasets

The project includes three pre-loaded datasets in the `data/` directory:

- `tiny_shakespeare.txt`: Shakespeare's works (default)
- `harry_potter.txt`: Harry Potter text
- `avatar_script.txt`: Avatar script

You can add your own text files to the `data/` directory and use them with the `--dataset` parameter.

## Model Architecture

The GPT implementation includes:

- **Multi-head Self-Attention**: Configurable number of attention heads
- **Positional Encoding**: Precomputed (or learned) positional embeddings
- **Layer Normalization**: Applied before attention and feed-forward layers
- **Dropout**: Configurable dropout for regularization
- **Feed-forward Networks**: Two-layer MLPs with GELU activation

## Examples

### Quick Start - Train a Small Model

```bash
# Train a small model on Shakespeare data
uv run training --epochs 1000 --ctx 64 --emb 128 --heads 8 --blocks 2 --generate 500
```

### Train a Larger Model

```bash
# Train a more capable model
uv run training --epochs 5000 --ctx 256 --emb 512 --heads 16 --blocks 6 --batch-size 32 --learning-rate 0.0003
```

### Generate Creative Text

```bash
# Generate a long passage
uv run generation 2000 --load your_trained_model.pt --seed 123
```

## Model Checkpoints

Trained models are automatically saved in the `checkpoints/` directory with timestamps. The naming convention includes key hyperparameters:

```
gpt.c{context_length}.e{embedding_dim}.h{num_heads}.b{num_blocks}-{timestamp}.pt
```

## Development

This project uses `ruff` for linting. To run the linter:

```bash
uv sync --dev
uv run ruff check
uv run ruff format
```

## Project Structure

```
├── mygpt/                  # Main package
│   ├── attention.py        # Attention mechanism implementation
│   ├── transformer.py      # Transformer block implementation
│   ├── gpt.py             # GPT facade model
│   ├── config.py          # Configuration management
│   ├── data_processor.py  # Data loading and processing
│   ├── tokenizers.py      # Text tokenization
│   └── entrypoints/       # Command-line interfaces
│       ├── train.py       # Training script
│       └── generate.py    # Generation script
├── data/                  # Training datasets
├── checkpoints/           # Saved model checkpoints
└── pyproject.toml        # Project configuration
```

## License

This project is open source and available under the MIT License.
