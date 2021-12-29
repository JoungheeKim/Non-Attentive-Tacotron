import torch
import os
from tqdm import tqdm
import re
import logging
from typing import Any, List, Optional
import unicodedata
from tacotron.utils import get_abspath
import string


class BaseTokenizer(object):
    """
        Tokenizer
        Only support
        - Number
        - English
        - Korean
        - Special Tokens [,.~?!]
    """
    special_regex = re.compile('[%s]' % re.escape(string.punctuation))
    remove_regex = re.compile('[%s]' % re.escape('"#$%&()*+-/:;<=>@[\\]^_{}”“'))

    tokenizer_save_name = 'pretrained_tokenizer.bin'
    special_pad_token = '<pad>'
    pad_token = '<pad>'
    normalize_option = 'NFKD'

    selected_token = {
        '’': "'",
        ",": ","
    }

    def __init__(self, vocabs={}, special_vocabs={}, normalize_option='NFKD'):
        super(BaseTokenizer, self).__init__()
        self.vocabs_to_ids = vocabs
        self.special_vocabs_to_ids = special_vocabs

        self.vocabs_to_ids[self.pad_token] = len(self.vocabs_to_ids)
        self.special_vocabs_to_ids[self.special_pad_token] = len(self.special_vocabs_to_ids)

        self.__rebuild()

        self.pad_id = self.vocabs_to_ids[self.pad_token]
        self.special_pad_id = self.special_vocabs_to_ids[self.special_pad_token]

        ## add space toekn
        self.add_token(' ')

        self.normalize_option = normalize_option
        NORMALIZE_OPTIONS = ['NFC', 'NFKC', 'NFD', 'NFKD']
        assert self.normalize_option in NORMALIZE_OPTIONS, 'normalize option should either {}. ex) Korean for "NFKD", English for "NFC"'.format(', '.join(NORMALIZE_OPTIONS))


    def __rebuild(self):
        self.ids_to_vocabs = [item for item in self.vocabs_to_ids.keys()]
        self.special_ids_to_vocabs = [item for item in self.special_vocabs_to_ids.keys()]
        self.vocabs_to_ids = {item:idx for idx, item in enumerate(self.ids_to_vocabs)}
        self.special_vocabs_to_ids = {item:idx for idx, item in enumerate(self.special_ids_to_vocabs)}

    def add_token(self, temp_string):
        assert type(temp_string) == str, 'must be string'
        temp_string = unicodedata.normalize(self.normalize_option, temp_string.lower())
        for temp_char in list(temp_string):
            if self.special_regex.search(temp_char):
                if temp_char not in self.special_vocabs_to_ids:
                    self.special_vocabs_to_ids[temp_char] = len(self.special_vocabs_to_ids)
                    self.special_ids_to_vocabs.append(temp_char)
            else:
                if temp_char not in self.vocabs_to_ids:
                    self.vocabs_to_ids[temp_char] = len(self.vocabs_to_ids)
                    self.ids_to_vocabs.append(temp_char)


    @classmethod
    def from_pretrained(cls, pretrained_path:str):
        pretrained_path = get_abspath(pretrained_path)

        tokenizer_path = os.path.join(pretrained_path, cls.tokenizer_save_name)
        logging.info('load tokenizer from [{}]'.format(pretrained_path))

        state = torch.load(tokenizer_path)
        normalize_option = state['normalize_option']
        vocabs = state['vocabs']
        special_vocabs = state['special_vocabs']

        return cls(vocabs, special_vocabs, normalize_option)

    @classmethod
    def build_tokenizer(cls, phones:List[list], normalize_option='NFKD'):
        tokenizer = cls({}, {}, normalize_option)

        for phone in tqdm(phones, desc='build tokenizer'):
            for p in phone:
                tokenizer.add_token(p)

        logging.info('bulild tokenizer with data length [{}]'.format(len(phones)))
        return tokenizer

    def save_pretrained(self, save_path:str):
        save_path = get_abspath(save_path)
        os.makedirs(save_path, exist_ok=True)

        tokenizer_path = os.path.join(save_path, self.tokenizer_save_name)
        torch.save(
            {
                'vocabs' : self.vocabs_to_ids,
                'special_vocabs' : self.special_vocabs_to_ids,
                'normalize_option': self.normalize_option,
             }, tokenizer_path
        )
        logging.info('save tokenizer to [{}]'.format(save_path))

    def encode(self, line:str, pace=None, **kwargs):
        assert type(line)==str, 'must be string'
        if pace is None:
            pace = 1

        ## convert german accents
        for before, after in self.selected_token.items():
            line = line.replace(before, after)

        line = self.remove_regex.sub('', line)
        line = unicodedata.normalize(self.normalize_option, line.lower())

        input_ids = list()
        special_input_ids = list()
        for temp_char in list(line):
            if self.special_regex.search(temp_char):
                if temp_char in self.special_vocabs_to_ids:
                    if len(special_input_ids) > 0:
                        special_input_ids[-1] = self.special_vocabs_to_ids[temp_char]
                    else:
                        input_ids.append(self.vocabs_to_ids[self.pad_token])
                        special_input_ids.append(self.special_vocabs_to_ids[temp_char])
            else:
                if temp_char in self.vocabs_to_ids:
                    input_ids.append(self.vocabs_to_ids[temp_char])
                    special_input_ids.append(self.special_vocabs_to_ids[self.special_pad_token])

        return {
            'input_ids' : input_ids,
            'special_input_ids' : special_input_ids,
            'pace_input_ids' : [pace] * len(input_ids),
        }

    def encode_pace(self, lines:dict):
        assert type(lines) == dict, 'must be dict contains string with pace'
        """
            ex) lines = {'I':2, ' love my ':1, 'children':2}
        """

        input_ids = list()
        special_input_ids = list()
        pace_input_ids = list()

        for sent, pace in lines.items():
            encoded_output = self.encode(sent, pace)
            input_ids.extend(encoded_output['input_ids'])
            special_input_ids.extend(encoded_output['special_input_ids'])
            pace_input_ids.extend(encoded_output['pace_input_ids'])

        return {
            'input_ids': input_ids,
            'special_input_ids': special_input_ids,
            'pace_input_ids': pace_input_ids,
        }


    def encode_batch(self, lines:List[str]):
        outputs=list()
        for line in tqdm(lines, desc='tokenizing'):
            outputs.append(self.encode(line))
        return outputs

    def decode(self, input_ids:list, special_input_ids:list=None, **kwargs):
        if special_input_ids is None:
            special_input_ids = [self.special_vocabs_to_ids[self.special_pad_token]] * len(input_ids)

        assert len(input_ids) == len(special_input_ids), 'must be same size'

        output_string = ''
        for input_id, special_input_id, in zip(input_ids, special_input_ids):
            if input_id != self.pad_id:
                output_string += self.ids_to_vocabs[input_id]
            if special_input_id != self.special_pad_id:
                output_string += self.special_ids_to_vocabs[special_input_id]

        return output_string

    def token_to_id(self, temp_str:str):
        return self.vocabs_to_ids(temp_str)

    def tokens_to_ids(self, tokens:List[str]):
        ids = list()
        for token in tokens:
            ids.append(self.vocabs_to_ids(token))
        return ids

    def get_num_labels(self):
        return len(self.vocabs_to_ids)

    def get_num_special_labels(self):
        return len(self.special_vocabs_to_ids)
