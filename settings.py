import os
import math


batch_limit = None

# Tokenizer settings
vocab_size = 11000

# Model parameters
embedding_dimensions = 512
num_layers = 8
number_heads = 8
head_size = 64

# Training parameters
batch_size = 16
block_size = 256

lr_max = 3e-4
lr_min = 3e-6
lr_start = 3e-5
warmup = 500
period = 50 * warmup

# YaRN config
YaRN_alpha = 2

# Path for Tokenizer and Checkpoint
vocab_path = "vocab.json"
checkpoint_path = os.path.join('checkpoints', 'spider.checkpoint')

# Learning rate Scheduler Cosine functions
def variable_lr(step):
    if step < warmup:
        return lr_start + (lr_max - lr_start)* step / warmup
    if step < warmup + period:
        return lr_min + (lr_max - lr_min) * math.cos((step - warmup)+math.pi/2/period) ** 4
    return lr_min