import os
import youtokentome as yttm
from typing import List


class BPETokenizer:
    
    def __init__(self, 
                model_file: str,
                bos_token='<BOS>', 
                eos_token='<EOS>',
                unk_token='<UNK>',
                pad_token='<PAD>',
                dropout_prob=0.0):
        self.model_file = model_file
        self.model_file = model_file
        self.bpe = self._read_model()
        self.vocab = self.bpe.vocab()
        self.rev_vocab = {self.vocab[i]: i for i in range(len(self.vocab))}
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.dropout_prob = dropout_prob
    
    def _read_model(self) -> None:
        if not os.path.isfile(self.model_file):
            raise Exception("model file doesn't exist, maybe you have to train it?")
        return yttm.BPE(model=self.model_file)
    
    @staticmethod
    def train_model(raw_text: List[str], vocab_size: int, model_file: str) -> None:
        train_data_path = 'temp.txt'
        with open(train_data_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(raw_text))
        yttm.BPE.train(data=train_data_path, vocab_size=vocab_size, model=model_file)
        os.remove(train_data_path)

    def tokenize(self, raw_text: str) -> List[str]:
        return self.bpe.encode([raw_text], output_type=yttm.OutputType.SUBWORD, dropout_prob=self.dropout_prob)[0]
    
    def tokenize_batch(self, batch: List[str]) -> List[List[str]]:
        tokenized = [self.tokenize(sent) for sent in batch]
        return tokenized
    
    def encode(self,  tokenized_str: List[str]) -> List[int]:
        encoded = []
        for token in tokenized_str:
            encoded.append(self.bpe.subword_to_id(token))
        return encoded

    def encode_batch(self, tokenized_batch: List[List[str]]) -> List[List[int]]:
        encoded_batch = [self.encode(tokens) for tokens in tokenized_batch]
        return encoded_batch

    def decode(self, encoded_str: List[int]) -> List[str]:
        decoded = []
        for id_ in encoded_str:
            decoded.append(self.bpe.id_to_subword(id_))
        return decoded

    def decode_batch(self, encoded_batch: List[List[int]]) -> List[List[str]]:
        decoded_batch = [self.decode(tokens) for tokens in encoded_batch]
        return decoded_batch
