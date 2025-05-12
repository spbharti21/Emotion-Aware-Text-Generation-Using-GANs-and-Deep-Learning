# config.py
import torch

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN       = 64
BATCH_SIZE    = 32        # increased from 16
EMBEDDING_DIM = 128
HIDDEN_DIM    = 256
NUM_EMOTIONS  = 28
LEARNING_RATE = 1e-4
EPOCHS        = 5         # increased from 1
