import math

import torch
import torch.utils.data as data



class WikiTextDataset(data.Dataset):
    def __init__(self, 
               text,
               pad_len,
               tokenizer,
               bos_token='<BOS>', 
               eos_token='<EOS>',
               unk_token='<UNK>',
               pad_token='<PAD>'):
        self.text = text
        self.pad_len = pad_len
        self.tokenizer = tokenizer
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token

    def __len__(self):
        return math.ceil(len(self.text) / self.pad_len)

    def __getitem__(self, ind):
        raw_txt = self.text[ind * (self.pad_len - 2) : (ind + 1) * (self.pad_len - 2)]
        # raw_txt = self.tokenizer.tokenize(raw_txt)
        txt = [self.bos_token,] + raw_txt + [self.eos_token,] + [self.pad_token] * (self.pad_len - len(raw_txt) - 2)
        loss_mask = ([1] * (len(raw_txt) + 2)) + ([0] * (self.pad_len - len(raw_txt) - 2))
        label = raw_txt + [self.eos_token] + [self.pad_token] * (self.pad_len - len(raw_txt) - 1)

        txt = self.tokenizer.encode(txt)
        label = self.tokenizer.encode(label)

        txt = torch.LongTensor(txt)
        label = torch.LongTensor(label)
        loss_mask = torch.LongTensor(loss_mask)
        assert len(txt) == len(label) == self.pad_len == len(loss_mask)
        return {
            'text': txt,
            'lm_label': label, 
            'loss_mask': loss_mask
        }
