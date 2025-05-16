import torch
from transformer import Transformer
from save import load_model
from tokenizer import decoder, encoder, tokenGen
from device import device
from WordPiece_tokenizer import tokenizer

model = Transformer()
model = model.to(device)

load_model(model)
model.eval()

vocab = tokenizer()
vocab.token_gen()

sample_text = "brother may I have"

encoded_sample = encoder(sample_text.lower())
encoded_sample = torch.tensor(encoded_sample, dtype= torch.long).unsqueeze(0)

output_sample = model.generate(encoded_sample, 100)[0]
output_sample = vocab.decoder(output_sample.tolist())

print(output_sample)