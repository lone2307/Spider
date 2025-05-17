from settings import *
from WordPiece import tokenizer

new_vocab = tokenizer()

new_vocab.load_vocab()

sample_text = "What work's, my countrymen, in hand? where go you. With bats and clubs?"

test = new_vocab.encoder(sample_text)

print("Sample:  " + sample_text)

print(f"Encoded: {test}")

print("Decoded: " + new_vocab.decoder(test))