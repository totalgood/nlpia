""" Translate documents in some way, like `sed`, only a bit more complex """
import os
import requests
import re
import json

import nltk
import spacy

from pugnlp.futil import find_files

from .constants import secrets, DATA_PATH


def minify_urls(filepath, ext='asc', url_regex=None, output_ext='.urls_minified', access_token=None):
    """ Use bitly or similar minifier to shrink all URLs in text files within a folder structure.

    Used for the NLPIA manuscript directory for Manning Publishing

    bitly API: https://dev.bitly.com/links.html

    Args:
      path (str): Directory or file path
      ext (str): File name extension to filter text files by. default='.asc'
      output_ext (str): Extension to append to filenames of altered files default='' (in-place replacement of URLs)

    FIXME: NotImplementedError! Untested!
    """
    access_token = access_token or secrets.bitly.access_token
    output_ext = output_ext or ''
    url_regex = re.compile(url_regex) if isinstance(url_regex, str) else url_regex
    filemetas = []
    for filemeta in find_files(filepath, ext=ext):
        filemetas += [filemeta]
        altered_text = ''
        with open(filemeta['path'], 'rt') as fin:
            text = fin.read()
        end = 0
        for match in url_regex.finditer(text):
            url = match.group()
            start = match.start()
            altered_text += text[:start]
            resp = requests.get('https://api-ssl.bitly.com/v3/shorten?access_token={}&longUrl={}'.format(
                access_token, url))
            js = resp.json()
            short_url = js['shortUrl']
            altered_text += short_url
            end = start + len(url)
        altered_text += text[end:]
        with open(filemeta['path'] + (output_ext or ''), 'wt') as fout:
            fout.write(altered_text)
    return altered_text


class TokenNormalizer:

    def __init__(self, mapping=None):
        self.mapping = {}
        if mapping is None or (isinstance(mapping, str) and os.path.isfile(mapping)):
            self.mapping = self.read_mapping(mapping)
        elif hasattr(mapping, 'get') and hasattr(mapping, '__getitem__'):
            self.mapping = mapping

    def read_mapping(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(DATA_PATH, 'emnlp_dict.txt')
        reg = re.compile("^([^\t\n]+)\t([^\t\n]+)\n$")
        result = {}
        with open(file_path) as f:
            for line in f:
                m = reg.match(line)
                if m is not None:
                    result[m.group(1)] = m.group(2)
                else:
                    print('WARN: TokenNormalizer.read_mapping() skipped: {}'.format(repr(line)))
        return result

    def normalize(self, word):
        if word in self.dict:
            return self.dict[word]
        return word


def clean_asciidoc(text):
    """ Transform asciidoc formatted text into ASCII text that NL parsers can hangle

    TODO:
      Tag lines and words with meta data like italics, underlined, bold, title, heading 1, etc
    """
    text = re.sub(r'(\b|^)[[_*]{1,2}([a-zA-Z0-9])', r'"\2', text)
    text = re.sub(r'([a-zA-Z0-9])[]_*]{1,2}', r'\1"', text)
    return text


def clean_markdown(text):
    return clean_asciidoc(text)


def split_sentences_nltk(text, language_model='tokenizers/punkt/english.pickle'):
    sentence_detector = nltk.data.load(language_model)
    return list(sentence_detector.tokenize(text.strip()))


def split_sentences_spacy(text, language_model='en'):
    """ You must download a spacy language model with python -m download 'en' """
    try:
        nlp = spacy.load(language_model)
    except (OSError, IOError):
        spacy.download(language_model)
    parsed_text = nlp(text)
    sentences = []
    for w, span in enumerate(parsed_text.sents):
        sent = ''.join(parsed_text[i].string for i in range(span.start, span.end)).strip()
        if len(sent):
            sentences.append(sent)
    return sentences


def segment_sentences(text, ext='asc', splitter=split_sentences_nltk):
    """ Return a list of all sentences and empty lines.

    TODO:
        1. process each line with an agressive sentence segmenter, like DetectorMorse
        2. process our manuscript to create a complete-sentence and heading training set normalized/simplified
           syntax net tree is the input feature set common words and N-grams inserted with their label as additional feature
        3. process a training set with a grammar checker and sentax next to bootstrap a "complete sentence" labeler.
        4. process each 1-3 line window (breaking on empty lines) with syntax net to label them
        5. label each 1-3-line window of lines as "complete sentence, partial sentence/phrase, or multi-sentence"

    >>> segment_se
    """
    sentences = []
    for filemeta in find_files(filepath, ext=ext):
        with open(filemeta['path'], 'rt') as fin:
            batch = []
            for i, line in enumerate(fin):
                if not line.strip():
                    sentences.extend(splitter('\n'.join(batch)))
                    batch = [line]  # may contain all whitespace
                else:
                    batch.append(line)
            if len(batch):
                sentences.extend(splitter('\n'.join(batch)))  # TODO: tag sentences with line + filename where they started
    return sentences


def fix_hunspell_json(badjson_path='en_us.json', goodjson_path='en_us_fixed.json'):
    """Fix the invalid hunspellToJSON.py json format by inserting double-quotes in list of affix strings

    Args:
      badjson_path (str): path to input json file that doesn't properly quote
      goodjson_path (str): path to output json file with properly quoted strings in list of affixes

    Returns:
      list of all words with all possible affixes in *.txt format (simplified .dic format)

    References:
      Syed Faisal Ali 's Hunspell dic parser: https://github.com/SyedFaisalAli/HunspellToJSON
    """
    with open(badjson_path, 'r') as fin:
        with open(goodjson_path, 'w') as fout:
            for i, line in enumerate(fin):
                line2 = re.sub(r'\[(\w)', r'["\1', line)
                line2 = re.sub(r'(\w)\]', r'\1"]', line2)
                line2 = re.sub(r'(\w),(\w)', r'\1","\2', line2)
                fout.write(line2)

    with open(goodjson_path, 'r') as fin:
        words = []
        with open(goodjson_path + '.txt', 'w') as fout:
            hunspell = json.load(fin)
            for word, affixes in hunspell['words'].items():
                words += [word]
                fout.write(word + '\n')
                for affix in affixes:
                    words += [affix]
                    fout.write(affix + '\n')

    return words
