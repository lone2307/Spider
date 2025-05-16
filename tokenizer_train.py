from settings import *
from new_tokenizer import tokenizer

new_vocab = tokenizer()

new_vocab.token_gen()

new_vocab.load_vocab()

print(new_vocab.encoder("How's going on"))