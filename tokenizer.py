import torch as nn
import torch.utils as utils
import re
from collections import Counter
from settings import *

def tokenGen():
    with open("archive/input.txt", "r", encoding="utf-8") as f:
        text = f.read().lower()

    tokens = re.findall(r"\b\w+\b|[.,!?;:]", text)

    word_freq = Counter(tokens)

    vocab = {word: i for i, (word, _) in enumerate(word_freq.most_common(30000), start=0)}
    return vocab

def encoder(text, vocab):
    tokens = re.findall(r"\b\w+\b|[.,!?;]", text) 
    return [vocab.get(token) for token in tokens]

def decoder(reverse_vocab, token_ids):
    return " ".join([reverse_vocab.get(i, "<unk>") for i in token_ids])