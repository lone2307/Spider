import re
from collections import Counter, OrderedDict
from settings import *


def vocab_gen():
    with open("archive/input.txt", "r", encoding="utf-8") as f:
        text = f.read().lower()
    
    return text

class tokenizer:
    def __init__(self):
        self.wrd_counter = Counter()
        self.text = vocab_gen()
        self.vocab = set()
        self.vocab_set = Counter()
        
    def token_gen(self):
        # gen and count word
        tokens = re.findall(r"\b\w+\b|[.,!?;:]", self.text)
        tokens = Counter(tokens)

        #divide word
        corpus_set = []
        for word, freq in tokens.items():
            word_set = [word[0]] + ["##" + c for c in word[1:]]
            word_set.append(freq)
            corpus_set.append(word_set)
        
        pre_vocab_set = set()
        
        while len(self.vocab) < vocab_size:
            # generate paired subword
            temp_counter = Counter()
            for word in corpus_set:
                for vocab in range(len(word[:-2])):
                    temp_counter[(word[vocab], word[vocab+1])] += word[-1]
            
            # get most common paired subword
            queue_vocab = temp_counter.most_common(1)
            queue_subword = queue_vocab[0][0][0] + queue_vocab[0][0][1][2:]
            queue_freq = queue_vocab[0][1]
            print((queue_subword, queue_freq))

            # change most paired subwords into most common subword
            self.vocab.add((queue_subword, queue_freq))
            for word in range(len(corpus_set)):
                temp_len = len(corpus_set[word][:-2])
                for vocab in range(temp_len):
                    if vocab + 2 == len(corpus_set[word]):
                        break
                    if corpus_set[word][vocab] + corpus_set[word][vocab+1][2:] == queue_subword:
                        corpus_set[word] = merge_pair(corpus_set[word], vocab)
                    if(vocab == len(corpus_set[word][:-2])):
                        break
            
            print(self.vocab)

        self.vocab = prune_redundant_subwords(self.vocab)
        self.vocab_set = sorted(self.vocab, key=lambda x: -x[1])[:vocab_size]                
    

        
    def decoder(self):
        return 0

    def encoder(self):
        return 0


def merge_pair(word_list, i):
    return word_list[:i] + [word_list[i] + word_list[i+1][2:]] + word_list[i+2:]

def prune_redundant_subwords(vocab):
    vocab_list = list(vocab)
    
    vocab_list.sort(key=lambda x: (-len(x[0]), -x[1]))

    filtered_vocab = []
    seen = {}

    for subword, freq in vocab_list:
        should_keep = True
        for kept_word, kept_freq in seen.items():
            if kept_word.startswith(subword) and kept_freq == freq:
                should_keep = False
                break
        if should_keep:
            filtered_vocab.append((subword, freq))
            seen[subword] = freq

    return set(filtered_vocab)