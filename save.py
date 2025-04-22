import os
import torch

from settings import *


def save_model(model):
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    
    if os.path.exists(checkpoint_path):
        os.rename(checkpoint_path, checkpoint_path + ".old")
    
    checkpoint = {
        "model": model.state_dict(),
    }
    
    torch.save(checkpoint, checkpoint_path)

def load_model(model):
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint["model"])