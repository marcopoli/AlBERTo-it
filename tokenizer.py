# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team,
# and Marco Polignano.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenization classes for Italian AlBERTo models."""
import collections
import logging
import os
import re
import logger

try:
    from ekphrasis.classes.preprocessor import TextPreProcessor
    from ekphrasis.classes.tokenizer import SocialTokenizer
    from ekphrasis.dicts.emoticons import emoticons
except ImportError:
    logger.warning(
        "You need to install ekphrasis to use AlBERToTokenizer"
        "pip install ekphrasis"
    )
    from pip._internal import main as pip
    pip(['install', '--user', 'ekphrasis'])
    from ekphrasis.classes.preprocessor import TextPreProcessor
    from ekphrasis.classes.tokenizer import SocialTokenizer
    from ekphrasis.dicts.emoticons import emoticons

try:
    import numpy as np
except ImportError:
    logger.warning(
        "You need to install numpy to use AlBERToTokenizer"
        "pip install numpy"
    )
    from pip._internal import main as pip
    pip(['install', '--user', 'pandas'])
    import pandas as pd

try:
    from transformers import BertTokenizer, WordpieceTokenizer
    from transformers.tokenization_bert import load_vocab
except ImportError:
    logger.warning(
        "You need to install pytorch-transformers to use AlBERToTokenizer"
        "pip install pytorch-transformers"
    )
    from pip._internal import main as pip
    pip(['install', '--user', 'pytorch-transformers'])
    from transformers import BertTokenizer, WordpieceTokenizer
    from transformers.tokenization_bert import load_vocab

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'user', 'percent', 'money', 'phone', 'time', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag"},
    fix_html=True,  # fix HTML tokens

    unpack_hashtags=True,  # perform word segmentation on hashtags

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)

class AlBERTo_Preprocessing(object):
    def __init__(self, do_lower_case=True, **kwargs):
        self.do_lower_case = do_lower_case

    def preprocess(self, text):
        if self.do_lower_case:
            text = text.lower()
        text = str(" ".join(text_processor.pre_process_doc(text)))
        text = re.sub(r'[^a-zA-ZÀ-ú</>!?♥♡\s\U00010000-\U0010ffff]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
        text = re.sub(r'^\s', '', text)
        text = re.sub(r'\s$', '', text)
        return text

class AlBERToTokenizer(BertTokenizer):

    def __init__(self, vocab_file, do_lower_case=True,
                 do_basic_tokenize=True, do_char_tokenize=False, do_wordpiece_tokenize=False, do_preprocessing = True, unk_token='[UNK]',
                 sep_token='[SEP]',
                 pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]', **kwargs):
        super(BertTokenizer, self).__init__(
            unk_token=unk_token, sep_token=sep_token, pad_token=pad_token,
            cls_token=cls_token, mask_token=mask_token, **kwargs)

        self.do_wordpiece_tokenize = do_wordpiece_tokenize
        self.do_lower_case = do_lower_case
        self.vocab_file = vocab_file
        self.do_basic_tokenize = do_basic_tokenize
        self.do_char_tokenize = do_char_tokenize
        self.unk_token = unk_token
        self.do_preprocessing = do_preprocessing

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file))

        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

        if do_wordpiece_tokenize:
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab,
                                                          unk_token=self.unk_token)
            
        self.base_bert_tok = BertTokenizer(vocab_file=self.vocab_file, do_lower_case=do_lower_case,
                                      unk_token=unk_token, sep_token=sep_token, pad_token=pad_token,
                                      cls_token=cls_token, mask_token=mask_token, **kwargs)

    def _convert_token_to_id(self, token):
        """Converts a token (str/unicode) to an id using the vocab."""
        # if token[:2] == '##':
        #     token = token[2:]

        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def convert_token_to_id(self, token):
        return self._convert_token_to_id(token)

        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, id):
        # if token[:2] == '##':
        #     token = token[2:]

        return list(self.vocab.keys())[int(id)]
    def convert_id_to_token(self, id):
        return self._convert_id_to_token(id)

    def _convert_tokens_to_string(self,tokens):
        """Converts a sequence of tokens (string) to a single string."""
        out_string = ' '.join(tokens).replace('##', '').strip()
        return out_string

    def convert_tokens_to_string(self,tokens):
        return self._convert_tokens_to_string(tokens)

    def _tokenize(self, text, never_split=None, **kwargs):
        if self.do_preprocessing:
            if self.do_lower_case:
                text = text.lower()
            text = str(" ".join(text_processor.pre_process_doc(text)))
            text = re.sub(r'[^a-zA-ZÀ-ú</>!?♥♡\s\U00010000-\U0010ffff]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
            text = re.sub(r'^\s', '', text)
            text = re.sub(r'\s$', '', text)
            # print(s)

        split_tokens = [text]
        if self.do_wordpiece_tokenize:
            wordpiece_tokenizer = WordpieceTokenizer(self.vocab,self.unk_token)
            split_tokens = wordpiece_tokenizer.tokenize(text)

        elif self.do_char_tokenize:
            tokenizer = CharacterTokenizer(self.vocab, self.unk_token)
            split_tokens = tokenizer.tokenize(text)

        elif self.do_basic_tokenize:
            """Tokenizes a piece of text."""
            split_tokens = self.base_bert_tok.tokenize(text)

        return split_tokens

    def tokenize(self, text, never_split=None, **kwargs):
        return self._tokenize(text, never_split)


class CharacterTokenizer(object):
    """Runs Character tokenziation."""

    def __init__(self, vocab, unk_token,
                 max_input_chars_per_word=100, with_markers=True):
        """Constructs a CharacterTokenizer.
        Args:
            vocab: Vocabulary object.
            unk_token: A special symbol for out-of-vocabulary token.
            with_markers: If True, "#" is appended to each output character except the
                first one.
        """
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.with_markers = with_markers

    def tokenize(self, text):
        """Tokenizes a piece of text into characters.

        For example:
            input = "apple"
            output = ["a", "##p", "##p", "##l", "##e"]  (if self.with_markers is True)
            output = ["a", "p", "p", "l", "e"]          (if self.with_markers is False)
        Args:
            text: A single token or whitespace separated tokens.
                This should have already been passed through `BasicTokenizer`.
        Returns:
            A list of characters.
        """

        output_tokens = []
        for i, char in enumerate(text):
            if char not in self.vocab:
                output_tokens.append(self.unk_token)
                continue

            if self.with_markers and i != 0:
                output_tokens.append('##' + char)
            else:
                output_tokens.append(char)

        return output_tokens

if __name__== "__main__":
    a = AlBERTo_Preprocessing(do_lower_case=True)
    s = "#IlGOverno presenta le linee guida sulla scuola #labuonascuola - http://t.co/SYS1T9QmQN"
    b = a.preprocess(s)
    print(b)

    c =AlBERToTokenizer(do_lower_case=True,vocab_file="vocab.txt", do_preprocessing=True)
    d = c.tokenize(s)
    print(d)