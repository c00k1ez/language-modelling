import nltk
from nltk.tokenize import MWETokenizer
from collections import Counter
import tqdm



class WordTokenizer:
    
    def __init__(self, 
                vocab_file=None, 
                bos_token='<bos>', 
                eos_token='<eos>',
                unk_token='<unk>',
                pad_token='<pad>'):
    
        self.vocab = None
        self.rev_vocab = None
        self.vocab_file = vocab_file
        self._read_vocab()
        self._rev_vocab()
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mwe_tokenizer = MWETokenizer([ ('<', 'bos', '>'),
                                            ('<', 'eos', '>'),
                                            ('<', 'unk', '>'),
                                            ('<', 'pad', '>')], separator='')

    def _write_vocab(self, file_name):
        write_file = open(file_name, 'w', encoding='utf-8')
        for num, word in enumerate(self.vocab):
            write_file.write('{}\t{}\n'.format(num, word))
        write_file.close()

    def _read_vocab(self):
        if self.vocab_file is None:
            print('Cannot read vocab from file. You have to build it')
        else:
            with open(self.vocab_file, 'r', encoding='utf-8') as read_file:
                self.vocab = []
                for d in read_file:
                    d = d.split('\t')
                    d[0] = int(d[0])
                    d[1] = d[1].replace('\n', '')
                    self.vocab.append(d[1])
            print('Read vocab file with {} tokens'.format(len(self.vocab)))
  
    def build_vocab(self, raw_text, vocab_path='vocab.txt', threshold=0.7):
        vocab_ = []
        for sent in tqdm.tqdm(raw_text):
            tokenized = nltk.word_tokenize(sent)
            tokenized = self.mwe_tokenizer.tokenize(tokenized)
            vocab_.extend(tokenized)
        cnt = Counter(vocab_)
        most_common_words = cnt.most_common(int(len(cnt) * threshold))
        most_common_words = [w[0] for w in most_common_words]
        most_common_words.remove('<unk>')
        vocab = []
        if self.bos_token is not None:
            vocab.append(self.bos_token)
        if self.eos_token is not None:
            vocab.append(self.eos_token)
        if self.unk_token is not None:
            vocab.append(self.unk_token)
        if self.pad_token is not None:
            vocab.append(self.pad_token)

        vocab.extend(most_common_words)
        self.vocab = vocab
        self._write_vocab(vocab_path)
        print('Build vocab with {} tokens'.format(len(self.vocab)))
        self._rev_vocab()

    def _rev_vocab(self):
        if self.vocab is None:
            print('Cannot build reverse vocab without vocab')
        else:
            self.rev_vocab = {token: num for num, token in enumerate(self.vocab)}

    def tokenize(self, raw_text):
        if self.vocab is None:
            raise Exception('You cannot tokenize without vocab')
        else:
            tok = nltk.word_tokenize(raw_text)
            tok = self.mwe_tokenizer.tokenize(tok)
            for i in range(len(tok)):
                if tok[i] not in self.rev_vocab:
                    tok[i] = self.unk_token

            return tok
      
    def tokenize_batch(self, batch):
        tokenized = []
        for sent in batch:
            tokenized.append(self.tokenize(sent))
        return tokenized

    def encode(self,  tokenized_str):
        encoded = []
        for token in tokenized_str:
            encoded.append(self.rev_vocab[token])
        return encoded

    def encode_batch(self, tokenized_batch):
        encoded_batch = []
        for tok in tokenized_batch:
            encoded_batch.append(self.encode(tok))
        return encoded_batch

    def decode(self, encoded_str):
        decoded = []
        for token in encoded_str:
            decoded.append(self.vocab[token])
        return decoded

    def decode_batch(self, encoded_batch):
        decoded_batch = []
        for tok in encoded_batch:
            decoded_batch.append(self.decode(tok))
        return decoded_batch