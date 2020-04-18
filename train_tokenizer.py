from framework.word_tokenizer import WordTokenizer
from framework.bpe_tokenizer import BPETokenizer
from framework.wikitext_parser import WikiTextParser



if __name__ == "__main__":

    parser = WikiTextParser('./data/wikitext-2')

    print('Succesfully load wikitext-2 data')

    tokenizer = WordTokenizer()
    
    all_data = []
    for key in list(parser.raw_sentencies.keys()):
        data = parser.raw_sentencies[key]
        all_data.extend(data)
    
    tokenizer.build_vocab(all_data, 'data/vocab_lower.txt')

    vocab_size = 15000
    print('-------------------')
    print('train bpe with {} vocab size'.format(vocab_size))
    model_name = './data/bpe_{}_vocab_lower.model'.format(vocab_size)
    tokenizer = BPETokenizer.train_model(all_data, vocab_size, model_name)

    print('test tokenizer')
    test_sent = 'i love cats'
    tokenizer = BPETokenizer(model_name)
    tok = tokenizer.tokenize(test_sent)
    print(tok)
    enc = tokenizer.encode(tok)
    print(enc)
    dec = tokenizer.decode(enc)
    print(dec)