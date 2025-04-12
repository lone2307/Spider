import os
import time
import torch
import wandb

from device import device

import torch.nn as nn
import torch


class Trainer:
    def __init__(self, criterion, optimizer):
        self.criterion = criterion
        self.optim = optimizer

    def train(self, model, dataloader, num_epochs):
        model.train()

        
        for epoch in range(num_epochs):
            for batch, (x, y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)
                
                self.optim.zero_grad()
                
                batch_output = model(x)
                
                loss = self.criterion(batch_output, y)
                loss.backward()
                
                if epoch * len(dataloader) + batch > 3000:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                self.optim.step()
                
                print(f'Epoch: {epoch} Batch: {batch} Loss: {(loss.item())}')