import torch
from transformer import Transformer
from settings import *
from device import device
from torch.utils.data import DataLoader
from trainer import Trainer
from data_loader import dataLoad
from param import get_param

model = Transformer()
model = model.to(device)

print(get_param(model))

def criterion(output, expected):
    return torch.nn.functional.cross_entropy(output.reshape(-1, vocab_size), expected.reshape(-1))

optimizer = torch.optim.AdamW(model.parameters(), lr=1, fused=True)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, variable_lr)

trainer = Trainer(criterion, optimizer, scheduler)

dataloader  = DataLoader(dataLoad(True), batch_size, True)
trainer.train(model, dataloader, 2, batch_limit)