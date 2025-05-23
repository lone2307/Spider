from torch.utils.data import Dataset
from settings import *
import torch
from WordPiece import tokenizer

class Dataset(Dataset):
    def __init__(self, text_index):
        self.text_index = text_index

    def __len__(self):
        return len(self.text_index) - block_size

    def __getitem__(self, idx):
        return self.text_index[idx:idx+block_size], self.text_index[idx+1:idx+block_size+1]


def dataLoad(train):
    tokenize = tokenizer()
    tokenize.token_gen()
    
    with open("archive/input.txt", "r", encoding="utf-8") as f:
        text = f.read().lower()

    encoded = tokenize.encoder(text)
    print(encoded)

    if train:
        encoded = encoded[:int(0.9*len(encoded))]
    else:
        encoded = encoded[int(0.9*len(encoded)):]
        
    return Dataset(torch.tensor(encoded))