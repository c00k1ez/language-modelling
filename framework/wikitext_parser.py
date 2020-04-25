from pathlib import Path
import nltk

from typing import Dict, List



class WikiTextParser:
    def __init__(self,
                dir_path: str, 
                train_file='wiki.train.tokens', 
                test_file='wiki.test.tokens',
                val_file='wiki.valid.tokens',
                lower_case=True):
        
        self.lower_case = lower_case
        self.train_file = train_file 
        self.test_file = test_file
        self.val_file = val_file
        self.dir_path = dir_path

        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        
        self.raw_sentencies = self._load_all_files_()
            
    def _load_all_files_(self) -> Dict[str, List]:

        raw_sentencies = {'train': [], 'test': [], 'valid': []}
        files = [self.train_file, self.test_file, self.val_file]

        for path, name in list(zip(files, list(raw_sentencies.keys()))):
            full_path = Path(self.dir_path + '/' + path)
            cleaned_data = self._read_file_(full_path)
            cleaned_data = self.sent_detector.tokenize(cleaned_data)
            cleaned_data = [sample for sample in cleaned_data if len(sample) > 5]
            raw_sentencies[name] = cleaned_data
            print('preprocess of {} file was ended'.format(name))
        
        total = 0
        print('---------------------------------')
        for key in raw_sentencies.keys():
            total += len(raw_sentencies[key])
            print('{} set: {} sentencies'.format(key, len(raw_sentencies[key])))
        print('total: {} sentencies'.format(total))
        print('---------------------------------')
        return raw_sentencies

    def _read_file_(self, full_path: str) -> str:
        data_file = open(full_path, 'r', encoding='utf-8')
        raw_str = []
        for s in data_file:
            s = s.replace('\t', '')
            s = s.replace('\n', '.')
            s = s.replace('..', '.')

            if len(s) <= 5:
                continue

            if self.lower_case is False:
                raw_str.append(s)
            else:
                raw_str.append(s.lower())
        raw_str = ' '.join(raw_str)
        return raw_str