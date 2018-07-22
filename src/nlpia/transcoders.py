""" Translate documents in some way, like `sed`, only a bit more complex """
import os
import requests
import regex
import json
from urllib.error import URLError

import nltk
import spacy

from pugnlp.futil import find_files
from nlpia.data_utils import iter_lines

from .constants import secrets, DATA_PATH

from logging import getLogger
logger = getLogger(__name__)


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
    url_regex = regex.compile(url_regex) if isinstance(url_regex, str) else url_regex
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
        reg = regex.compile("^([^\t\n]+)\t([^\t\n]+)\n$")
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
    """ Transform asciidoc text into ASCII text that NL parsers can handle

    TODO:
      Tag lines and words with meta data like italics, underlined, bold, title, heading 1, etc

    >>> clean_asciidoc('**Hello** _world_!')
    '"Hello" "world"!'
    """
    text = regex.sub(r'(\b|^)[[_*]{1,2}([a-zA-Z0-9])', r'"\2', text)
    text = regex.sub(r'([a-zA-Z0-9])[]_*]{1,2}', r'\1"', text)
    return text


def clean_markdown(text):
    r""" Transform markdown text into ASCII natural language text that NL parsers can handle

    >>> clean_markdown('**Hello** _world_!')
    '"Hello" "world"!'
    """
    return clean_asciidoc(text)


def split_text_blocks(text):
    r""" Splits asciidoc and markdown files into a list of lists (blocks text in a list of lines of text)

    >>> split_text_blocks("# Title \nHello world! \nI'm here today to\nstrip\nyour text.\n \t  \r\nNext block\n\nwas short.")
    [['# Title \n',
      'Hello world! \n',
      "I'm here today to\n",
      'strip\n',
      'your text.\n',
      ' \t  \r\n'],
     ['Next block\n', '\n'],
     ['was short.']]
    >> split_text_blocks(os.path.join(DATA_PATH, 'book'))
    """
    blocks = []
    block = []
    for line in iter_lines(text):
        block.append(line)
        if not line.strip():
            blocks.append(block)
            block = []
    if block:
        blocks.append(block)
    return blocks


def split_sentences_nltk(text, language_model='tokenizers/punkt/english.pickle'):
    try:
        sentence_detector = nltk.data.load(language_model)
    except LookupError:
        try:
            nltk.download('punkt', raise_on_error=True)
            sentence_detector = nltk.data.load(language_model)
        except ValueError:
            return split_sentences_regex(text)

    return list(sentence_detector.tokenize(text.strip()))


def split_sentences_regex(text):
    """ Use dead-simple regex to split text into sentences. Very poor accuracy.

    >>> split_sentences_regex("Hello World. I'm I.B.M.'s Watson. --Watson")
    ['Hello World.', "I'm I.B.M.'s Watson.", '--Watson']
    """
    parts = regex.split(r'([a-zA-Z0-9][.?!])[\s$]', text)
    sentences = [''.join(s) for s in zip(parts[0::2], parts[1::2])]
    return sentences + [parts[-1]] if len(parts) % 2 else sentences


def split_sentences_spacy(text, language_model='en'):
    r""" You must download a spacy language model with python -m download 'en'

    The default English language model for spacy tends to be a lot more agressive than NLTK's punkt:

    >>> split_sentences_nltk("Hi Ms. Lovelace.\nI'm a wanna-\nbe human @ I.B.M. ;) --Watson 2.0")
    ['Hi Ms. Lovelace.', "I'm a wanna-\nbe human @ I.B.M.", ';) --Watson 2.0']
    >>> split_sentences_spacy("Hi Ms. Lovelace.\nI'm a wanna-\nbe human @ I.B.M. ;) --Watson 2.0")
    ['Hi Ms. Lovelace.', "I'm a wanna-", 'be human @', 'I.B.M. ;) --Watson 2.0']

    >>> split_sentences_spacy("Hi Ms. Lovelace. I'm at I.B.M. --Watson 2.0")
    ['Hi Ms. Lovelace.', "I'm at I.B.M. --Watson 2.0"]
    >>> split_sentences_nltk("Hi Ms. Lovelace. I'm at I.B.M. --Watson 2.0")
    ['Hi Ms. Lovelace.', "I'm at I.B.M.", '--Watson 2.0']
    """
    try:
        nlp = spacy.load(language_model)
    except (OSError, IOError):
        try:
            spacy.cli.download(language_model)
        except URLError:
            logger.warn("Unable to download Spacy language model '{}'. Using offline NLTK punkt sentence splitter instead.")
            return split_sentences_nltk(text)
    parsed_text = nlp(text)
    sentences = []
    for w, span in enumerate(parsed_text.sents):
        sent = ''.join(parsed_text[i].string for i in range(span.start, span.end)).strip()
        if len(sent):
            sentences.append(sent)
    return sentences


def get_splitter(fun=None):
    if fun is None:
        fun = split_sentences_nltk
    elif fun in locals():
        fun = locals()[fun]
    elif fun.lower().endswith('nltk'):
        fun = split_sentences_nltk
    elif fun.lower().endswith('spacy'):
        fun = split_sentences_spacy

    try:
        fun('Test sentence.')
    except:  # noqa
        fun = None
    if callable(fun):

        return fun
    else:
        return None


def segment_sentences(path=os.path.join(DATA_PATH, 'book'), splitter=split_sentences_nltk, **find_files_kwargs):
    """ Return a list of all sentences and empty lines.

    TODO:
        1. process each line with an agressive sentence segmenter, like DetectorMorse
        2. process our manuscript to create a complete-sentence and heading training set normalized/simplified
           syntax net tree is the input feature set common words and N-grams inserted with their label as additional feature
        3. process a training set with a grammar checker and sentax next to bootstrap a "complete sentence" labeler.
        4. process each 1-3 line window (breaking on empty lines) with syntax net to label them
        5. label each 1-3-line window of lines as "complete sentence, partial sentence/phrase, or multi-sentence"

    >>> len(segment_sentences(path=os.path.join(DATA_PATH, 'book')))
    17319
    >>> len(segment_sentences(path=os.path.join(DATA_PATH, 'psychology-scripts.txt'), splitter=split_sentences_nltk))
    23
    """
    sentences = []
    if os.path.isdir(path):
        for filemeta in find_files(path, **find_files_kwargs):
            with open(filemeta['path']) as fin:
                i, batch = 0, []
                try:
                    for i, line in enumerate(fin):
                        if not line.strip():
                            sentences.extend(splitter('\n'.join(batch)))
                            batch = [line]  # may contain all whitespace
                        else:
                            batch.append(line)
                except (UnicodeDecodeError, IOError):
                    logger.error('UnicodeDecodeError or IOError on line {} in file {} from stat: {}'.format(
                        i + 1, fin.name, filemeta))
                    raise

                if len(batch):
                    # TODO: tag sentences with line + filename where they started
                    sentences.extend(splitter('\n'.join(batch)))
    else:
        batch = []
        for i, line in enumerate(iter_lines(path)):
            # TODO: filter out code and meta lines using asciidoc or markdown parser
            # split into batches based on empty lines
            if not line.strip():
                sentences.extend(splitter('\n'.join(batch)))
                # first line may contain all whitespace
                batch = [line]
            else:
                batch.append(line)
        if len(batch):
            # TODO: tag sentences with line + filename where they started
            sentences.extend(splitter('\n'.join(batch)))

    return sentences


def tag_code(line):
    cleaned = line.strip().lower()
    if cleaned.startswith('>>> ') or cleaned.startswith:
        return 'code.python.doctest'


def tag_code_lines(text, markup=None):
    if (markup is None or markup.lower() in ('asc', '.asc', 'adoc', '.adoc', '.asciidoc') or (
            os.path.isfile(text) and text.lower().split('.')[-1] in ('asc', 'adoc', 'asciidoc'))):
        markup = 'asciidoc'
    lines = []
    within_codeblock = False
    for i, line in enumerate(iter_lines(text)):
        # TODO: filter out code and meta lines using asciidoc or markdown parser
        # split into batches based on empty lines
        tag = tag_code(line, markup=markup)
        if within_codeblock or tag.startswith('code.'):
            if tag.endswith('.end'):
                within_codeblock = False
            elif tag == 'code.start':
                within_codeblock = True
        lines.append(line, tag)
    return lines


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
                line2 = regex.sub(r'\[(\w)', r'["\1', line)
                line2 = regex.sub(r'(\w)\]', r'\1"]', line2)
                line2 = regex.sub(r'(\w),(\w)', r'\1","\2', line2)
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
