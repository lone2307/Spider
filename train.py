import torch
from tokenizer import encoder, tokenGen
from transformer import Transformer
from settings import *
from device import device
from torch.utils.data import DataLoader
from trainer import Trainer
from data_loader import dataLoad

vocab = tokenGen()

sample_text = "What work's, my countrymen, in hand? where go you. With bats and clubs?"

encoded_sample = encoder(sample_text.lower(),vocab)

model = Transformer()
model = model.to(device)

def criterion(output, expected):
    return torch.nn.functional.cross_entropy(output.reshape(-1, vocab_size), expected.reshape(-1))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

trainer = Trainer(criterion, optimizer)

dataloader  = DataLoader(dataLoad(True), batch_size, True)
trainer.train(model, dataloader, 10)