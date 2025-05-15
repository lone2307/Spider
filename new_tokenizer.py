import torch
import numpy as np
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
            word_set = []
            word_set.append(word[0])
            for x in range(1,len(word)):
                word_set.append("##" + word[x])
            word_set.append(freq)
            corpus_set.append(word_set)
        
        pre_vocab_set = set()
        
        while(len(self.vocab) < vocab_size):
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
            
            # remove smaller subword
            for word, freq in self.vocab:
                if word in queue_vocab[0][0]:
                    if freq == queue_freq:
                        self.vocab.remove(word)

            # 
            self.vocab.add((queue_subword, queue_freq))
            for word in corpus_set:
                temp_len = len(word[:-2])
                for vocab in range(temp_len):
                    if(vocab == len(word[:-2])):
                        break
                    if word[vocab] + word[vocab+1] == queue_subword:
                        word = merge_pair(word, vocab)
                        if(vocab == len(word[:-2])):
                            break
            
            print(self.vocab)
        self.vocab_set = pre_vocab_set.most_common(vocab_size)
        
        print(pre_vocab_set)
                
    

        
    def decoder(self):
        return 0

    def encoder(self):
        return 0


def merge_pair(arr, i):
    if i < 0 or i >= len(arr) - 1:
        raise IndexError("Invalid merge index")
    return arr[:i] + [arr[i] + arr[i+1]] + arr[i+2:]