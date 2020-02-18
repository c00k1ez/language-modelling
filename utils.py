import torch

from torch.utils.data import DataLoader

from framework.datasets.wikitext_dataset import WikiTextDataset
from framework.wikitext_parser import WikiTextParser
from framework.word_tokenizer import WordTokenizer

import numpy as np
import random


def load_dataloaders(train_batch_size, test_batch_size, pad_len, vocab_file='./data/vocab.txt'):

    parser = WikiTextParser('./data/wikitext-2')

    tokenizer = WordTokenizer(vocab_file)

    loaders = {
        'train': DataLoader(WikiTextDataset(parser.raw_sentencies['train'], pad_len, tokenizer), 
                                            train_batch_size, shuffle=True),
        'test': DataLoader(WikiTextDataset(parser.raw_sentencies['test'], pad_len, tokenizer), 
                                            test_batch_size, shuffle=True),
        'valid': DataLoader(WikiTextDataset(parser.raw_sentencies['test'], pad_len, tokenizer), 
                                            test_batch_size, shuffle=True)
    }

    return loaders

def generate_sentence(md, tok, start_str):
    md.eval()
    with torch.no_grad():
        tokenized = tok.tokenize(start_str)
        tokenized_ = tok.encode(tokenized)
        gen = torch.exp(md(torch.LongTensor([tokenized_])))
        gen = torch.argmax(gen[0][-1]).item()
        while tok.decode([gen])[0] != '<eos>':
            tokenized_.append(gen)
            gen = torch.exp(md(torch.LongTensor([tokenized_])))
            gen = gen[0][-1].argmax().item()
    print(' '.join(tok.decode(tokenized_)))

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)