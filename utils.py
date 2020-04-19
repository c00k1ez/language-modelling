import torch

from torch.utils.data import DataLoader

from framework.datasets.wikitext_dataset import WikiTextDataset
from framework.wikitext_parser import WikiTextParser
from framework.word_tokenizer import WordTokenizer
from framework.bpe_tokenizer import BPETokenizer

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import CometLogger
from pytorch_lightning import LightningModule, Trainer

import numpy as np
import random

import os


def load_dataloaders(train_batch_size, 
                    test_batch_size, 
                    pad_len, 
                    vocab_file='./data/vocab.txt', 
                    tokenizer_type='word',
                    model_file=None,
                    dropout_prob=0.0):
    
    if tokenizer_type not in ['word', 'bpe']:
        raise ValueError("You have to use only 'word' or 'bpe' tokenizer type")

    if tokenizer_type == 'word':
        tokenizer = WordTokenizer(vocab_file)
    elif tokenizer_type == 'bpe':
        if model_file is None:
            raise ValueError("When you use 'bpe' tokenizer, you have to set path to BPE model")
        tokenizer = BPETokenizer(model_file, dropout_prob=dropout_prob)

    parser = WikiTextParser('./data/wikitext-2')

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
        while tok.decode([gen])[0] != '<EOS>':
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


def CustomModelCheckpoint(ModelCheckpoint):

    def __init__(self, model_name, **kwargs):
        super(CustomModelCheckpoint, self).__init__(**kwargs)
        self.model_name = model_name

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super(CustomModelCheckpoint, self).on_validation_end(trainer, pl_module)
        if isinstance(trainer.logger, CometLogger):
            path = self.dirpath + '/' + os.listdir(self.dirpath)[0]
            trainer.logger.experiment.log_model(self.model_name, path)
