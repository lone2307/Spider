from turtle import forward
import torch.nn as nn
import torch
import math
from settings import *


def YaRN_RoPE(x, seq_len, training = False):
    i = torch.arange(0, embedding_dimensions, 2, device=x.device)
    NTK_alpha = 1 if training else seq_len / max_seq_len
    inv_freq = 1.0 / (10000 ** (i / (embedding_dimensions * NTK_alpha)))
    pos = torch.arange(seq_len, device=x.device)
    yarn_scaling = (1/torch.sqrt(1 + YaRN_alpha * ((pos/block_size) ** 2 ))).unsqueeze(1)
    theta = torch.einsum("i,j->ij", pos, inv_freq)
    theta = theta * yarn_scaling


    sin = torch.sin(theta)[None, None, :, :]
    cos = torch.cos(theta)[None, None, :, :]
    
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]

    x_even_after = x_even * cos - x_odd * sin
    x_odd_after  = x_even * sin + x_odd * cos

    x_rotated = torch.empty_like(x)
    x_rotated[...,0::2] = x_even_after
    x_rotated[...,1::2] = x_odd_after

    return x_rotated


# Old PE, not use
class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        
        pe = torch.zeros(max_seq_len,embedding_dimensions)

        for pos in range(max_seq_len):
            for i in range(0, embedding_dimensions, 2):
                pe[pos,i] = math.sin(pos / (10000 ** ((2 * i)/embedding_dimensions)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2*i)/embedding_dimensions)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        x = x * math.sqrt(embedding_dimensions)
        
        seq_len = x.size(1)
        
        x = x + self.pe[:,:seq_len]
        
        return x


class multiHeadAttention(nn.Module):
    def __init__ (self):
        super().__init__()

        self.key   = nn.Linear(embedding_dimensions, head_size, bias=False)
        self.query = nn.Linear(embedding_dimensions, head_size, bias=False)
        self.value = nn.Linear(embedding_dimensions, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.proj = nn.Linear(embedding_dimensions,embedding_dimensions)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        seq_len = x.size()[1]
        q = self.query(x).view(batch_size, seq_len, number_heads, head_size).permute(0,2,1,3)
        k = self.key(x).view(batch_size, seq_len, number_heads, head_size).permute(0,2,1,3)
        v = self.value(x).view(batch_size, seq_len, number_heads, head_size).permute(0,2,1,3)

        q = self.YaRN_RoPE(q)
        k = self.YaRN_RoPE(k)

        score = q @ k.transpose(-2,-1) * head_size**-0.5
        score = score.masked_fill(self.tril[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        score = nn.functional.softmax(score,dim=-1)
        
        out = score @ v
        out = out.contiguous().view(batch_size, seq_len, embedding_dimensions)
        out = self.dropout(self.proj(out))
        
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.feedForward = nn.Sequential(
            nn.Linear(embedding_dimensions, 4*embedding_dimensions),
            nn.ReLU(),
            nn.Linear(embedding_dimensions*4, embedding_dimensions),
            nn.Dropout(0.2)
        )    
    def forward(self,x):

        return self.feedForward(x)


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = multiHeadAttention()
        self.dropout = nn.Dropout(0.2)
        self.feedForward = FeedForward()
        
        self.norm1 = nn.LayerNorm(embedding_dimensions)
        self.norm2 = nn.LayerNorm(embedding_dimensions)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feedForward(self.norm2(x))

        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dimensions)
        self.block = nn.Sequential(*[DecoderBlock() for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_dimensions)
        self.head = nn.Linear(embedding_dimensions, vocab_size)
        
    def forward(self, idx):
        embedding = self.embeddings(idx)
        x = self.block(embedding)
        x = self.norm(x)
        logits = self.head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        device = next(self.parameters()).device  # Get the device model is on

        idx = idx.to(device)  # Ensure idx is on the same device as model

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx