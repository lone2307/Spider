import re
from collections import Counter
from settings import *
import string
import re
import json


def vocab_gen():
    with open("archive/input.txt", "r", encoding="utf-8") as f:
        text = f.read().lower()
    
    return text

class tokenizer:
    def __init__(self):
        self.text = vocab_gen()
        self.vocab = set()
        self.vocab_set = {}
        
    def token_gen(self):
        # gen and count word
        tokens = re.findall(r"\b\w+\b|[.,!?;:(){}\[\]\"'<>\\/@#$%^&*_+=|~`-]", self.text)
        punct_set = set(string.punctuation)
        word_tokens = [tok for tok in tokens if tok not in punct_set]
        tokens_counts = Counter(word_tokens)
        
        
        #remove word length 1
        for word in list(tokens_counts):
            if len(word) == 1:
                self.vocab.add((word, tokens_counts[word]))
                del tokens_counts[word]

        #divide word to subwords
        corpus_set = []
        for word, freq in tokens_counts.items():
            word_set = [word[0]] + ["##" + c for c in word[1:]]
            word_set.append(freq)
            corpus_set.append(word_set)
        
        while len(self.vocab) < vocab_size - 32:
            # generate paired subword
            temp_counter = Counter()
            for word in corpus_set:
                token = word[:-1]
                freq = word[-1]
                for vocab in range(len(token)-1):
                    pair = token[vocab], token[vocab+1]
                    temp_counter[pair] += freq
            
            # get most common paired subword
            queue_vocab = temp_counter.most_common(1)
            queue_subword = queue_vocab[0][0][0] + queue_vocab[0][0][1][2:]
            self.vocab.add((queue_subword, queue_vocab[0][1]))

            # combine all the most common pairs
            new_corpus = []

            for word in corpus_set:
                tokens = word[:-1]  # exclude frequency
                freq = word[-1]
                
                i = 0
                new_word = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] + tokens[i+1][2:] == queue_subword:
                        new_word.append(queue_subword)
                        i += 2
                    else:
                        new_word.append(tokens[i])
                        i += 1

                new_word.append(freq)
                new_corpus.append(new_word)

            corpus_set = new_corpus
            
            # remove unimportant vocab
            if len(self.vocab) == vocab_size:
                print("Cleaning up vocab....")
                self.vocab = prune_redundant_subwords(self.vocab)
                print(f"Erased {vocab_size - len(self.vocab)} unimportant vocab")

        print(len(self.vocab))
        # sorting vocab_set
        self.vocab_set = sorted(self.vocab, key=lambda x: -x[1])
        
        # add 32 punctuations
        for p in string.punctuation:
            self.vocab_set.append((p, 999))

        # save vocab
        self.save_vocab()

    def get_vocab_set(self):
        print(self.vocab_set)

    def save_vocab(self):
        token_to_id = {token: idx for idx, (token, _) in enumerate(self.vocab_set)}
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(token_to_id, f, ensure_ascii=False, indent=2)
            
    def load_vocab(self):
        with open(vocab_path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        
        print(token_to_id)
        
        inverse = sorted(token_to_id.items(), key=lambda x: len(x))
        self.vocab_set = [(token, id) for token, id in inverse]

    def encoder(self, text):
        # convert text to lowercase and split words + punctuation
        tokens = re.findall(r"\b\w+\b|[.,!?;:(){}\[\]\"'<>\\/@#$%^&*_+=|~`-]", text.lower())

        # convert vocab_set from list of (subword, freq) to just subword strings
        vocab_lookup = {subword for subword, _ in self.vocab_set}

        output = []

        for token in tokens:
            # exact match (e.g., punctuation or whole word in vocab)
            if token in vocab_lookup:
                output.append(token)
                continue

            # otherwise, apply WordPiece greedy matching
            i = 0
            while i < len(token):
                matched = False
                end = len(token)
                while end > i:
                    sub = token[i:end]
                    if i > 0:
                        sub = "##" + sub
                    if sub in vocab_lookup:
                        output.append(sub)
                        i = end
                        matched = True
                        break
                    end -= 1
                if not matched:
                    output.append("[UNK]")
                    break

        return output

    def decoder(self):
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