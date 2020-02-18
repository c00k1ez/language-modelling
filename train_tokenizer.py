from framework.word_tokenizer import WordTokenizer
from framework.wikitext_parser import WikiTextParser



if __name__ == "__main__":

    parser = WikiTextParser('./data/wikitext-2')

    print('Succesfully load wikitext-2 data')

    tokenizer = WordTokenizer()
    
    all_data = []
    for key in list(parser.raw_sentencies.keys()):
        data = parser.raw_sentencies[key]
        all_data.extend(data)
    
    tokenizer.build_vocab(all_data, 'data/vocab.txt')