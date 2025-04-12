import torch.nn as nn
import torch
import math
from settings import *

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


class singleHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.key   = nn.Linear(embedding_dimensions, head_size, bias=False)
        self.query = nn.Linear(embedding_dimensions, head_size, bias=False)
        self.value = nn.Linear(embedding_dimensions, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        b,t,c = x.shape
        k = self.key(x)
        q = self.key(x)
        v = self.value(x)

        score = q @ k.transpose(-2,-1) * c**-0.5
        score = score.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        score = nn.functional.softmax(score,dim=-1)
        
        out = score @ v
        
        return out


class multiHeadAttention(nn.Module):
    def __init__ (self):
        super().__init__()
                
        self.norm = nn.RMSNorm(embedding_dimensions)
        self.multiHead = nn.ModuleList([singleHeadAttention() for _ in range(number_heads)])
        self.proj = nn.Linear(embedding_dimensions,embedding_dimensions)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.multiHead], dim=-1)
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
        self.norm = nn.RMSNorm(embedding_dimensions)
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
        self.pos_embed = PositionalEncoding()
        
    def forward(self, idx):
        B, T = idx.shape
        
        embedding = self.embeddings(idx)
        embedding = self.pos_embed(embedding)
        x = self.block(embedding)
        x = self.norm(x)
        logits = self.head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx