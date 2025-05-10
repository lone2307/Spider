import time

from device import device

import torch.nn as nn
import torch
from save import save_model


class Trainer:
    def __init__(self, criterion, optimizer, scheduler):
        self.criterion = criterion
        self.optim = optimizer
        self.scheduler = scheduler

    def train(self, model, dataloader, num_epochs, limit = None):
        model.train()
        
        last_time = time.time()
        time_length = 30

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
                self.scheduler.step()
                
                print(f'Epoch: {epoch} Batch: {batch} Loss: {(loss.item())} Lr: {(self.scheduler.get_last_lr()[0])}')
                
                if time.time() - last_time > time_length:
                    last_time = time.time()
                    
                    print('Saving')
                    save_model(model)
                
                if batch == limit:
                    break