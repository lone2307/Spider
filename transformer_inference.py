import torch
from transformer import Transformer
from save import load_model
from tokenizer import decoder, encoder, tokenGen
from device import device

model = Transformer()
model = model.to(device)

load_model(model)
model.eval()

vocab = tokenGen()
reverse_vocab = {idx: word for word, idx in vocab.items()}

sample_text = "brother may I have"

encoded_sample = encoder(sample_text.lower(),vocab)
encoded_sample = torch.tensor(encoded_sample, dtype= torch.long).unsqueeze(0)

output_sample = model.generate(encoded_sample, 100)[0]
output_sample = decoder(reverse_vocab, output_sample.tolist())

print(output_sample)