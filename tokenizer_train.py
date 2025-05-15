
from settings import *
from new_tokenizer import tokenizer


new_vocab = tokenizer()

new_vocab.token_gen()

print(new_vocab.get_vocab_set)