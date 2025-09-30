from dataclasses import dataclass


@dataclass(frozen=True)
class Defaults:
    """Default program parameters"""

    # Model
    CTX_LEN = 64
    EMB_DIM = 128
    NUM_HEADS = 16
    NUM_BLOCKS = 1
    # Training
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    DROPOUT = 0.2
    EPOCHS = 5000
    L2_REG = 1e-2
    EVAL_INTERVAL = 500
    EVAL_ITERS = 200
    TRAIN_VAL_SPLIT = 0.9
    DATASET = "tiny_shakespeare.txt"
    # Inference
    MAX_TOKENS = 500
