#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Loaders and downloaders for data files and models required for the examples in NLP in Action

>>> df = get_data('cities_us')
>>> df.iloc[:3,:2]
        geonameid                           city
131484    4295856  Indian Hills Cherokee Section
137549    5322551                         Agoura
134468    4641562                         Midway

Google N-Gram Viewer data (at least the 1-grams) is available with get_data as well.
The smallest 1-gram table is for the "first letter" pos (part of speech tags all alone):
>>> df = get_data('1gram_pos')
>>> df
  term_pos  year  term_freq  book_freq
0    _ADP_  1505       3367          1
1    _ADP_  1507       4619          1
2    _ADP_  1515      37423          1
...

The words that start with X is also a pretty small list:
>>> df = get_data('1gram_x')
>>> df
        term_pos  year  term_freq  book_freq
0         X'rays  1914          1          1
1         X'rays  1917          1          1
2         X'rays  1919          1          1
3         X'rays  1921          1          1
...
[3929235 rows x 4 columns]
"""
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import (bytes, dict, int, list, object, range, str,  # noqa
    ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
from future import standard_library
standard_library.install_aliases()  # noqa
from past.builtins import basestring

# from traceback import format_exc
import os
import re
import json
import logging
import shutil
from traceback import format_exc
from zipfile import ZipFile
from math import ceil
from itertools import product, zip_longest
import requests
from requests.exceptions import ConnectionError, InvalidURL, InvalidSchema, InvalidHeader, MissingSchema
from urllib.parse import urlparse
from urllib.error import URLError
from lxml.html import fromstring as parse_html
from copy import deepcopy

import pandas as pd
import gzip
import tarfile
import ftplib

import spacy
from tqdm import tqdm
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from pugnlp.futil import mkdir_p, path_status, find_files
from pugnlp.util import clean_columns
from nlpia.constants import DATA_PATH, BIGDATA_PATH
from nlpia.constants import DATA_INFO_FILE, BIGDATA_INFO_FILE, BIGDATA_INFO_LATEST


_parse = None  # placeholder for SpaCy parser + language model

INT_MAX = INT64_MAX = 2 ** 63 - 1
INT_MIN = INT64_MIN = - 2 ** 63
INT_NAN = INT64_NAN = INT64_MIN
INT_MIN = INT64_MIN = INT64_MIN + 1
MIN_DATA_FILE_SIZE = 100  # get_data will fail on files < 100 bytes

np = pd.np
logger = logging.getLogger(__name__)
# logging.config.dictConfig(LOGGING_CONFIG)
# # doesn't display line number, etc
# if os.environ.get('DEBUG'):
#     logging.basicConfig(level=logging.DEBUG)


# SMALLDATA_URL = 'http://totalgood.org/static/data'

W2V_FILES = [
    'GoogleNews-vectors-negative300.bin.gz',
    'glove.6B.zip', 'glove.twitter.27B.zip', 'glove.42B.300d.zip', 'glove.840B.300d.zip',
]
# You probably want to rm nlpia/src/nlpia/data/bigdata_info.csv if you modify any of these so they don't overwrite what you do here
ZIP_FILES = {
    'GoogleNews-vectors-negative300.bin.gz': None,
    'glove.6B.zip': ['glove.6B.50d.w2v.txt', 'glove.6B.100d.w2v.txt', 'glove.6B.200d.w2v.txt', 'glove.6B.300d.w2v.txt'],
    'glove.twitter.27B.zip': None,
    'glove.42B.300d.zip': None,
    'glove.840B.300d.zip': None,
}
ZIP_PATHS = [[os.path.join(BIGDATA_PATH, fn) for fn in ZIP_FILES[k]] if ZIP_FILES[k] else k for k in ZIP_FILES.keys()]


def load_imdb_df(dirpath=os.path.join(BIGDATA_PATH, 'aclImdb'), subdirectories=(('train', 'test'), ('pos', 'neg', 'unsup'))):
    """ Walk directory tree starting at `path` to compile a DataFrame of movie review text labeled with their 1-10 star ratings

    Returns:
      DataFrame: columns=['url', 'rating', 'text'], index=MultiIndex(['train_test', 'pos_neg_unsup', 'id'])

    TODO:
      Make this more robust/general by allowing the subdirectories to be None and find all the subdirs containing txt files

    >> imdb_df().head()
                                                          url  rating                                               text
    index0 index1 index2
    train  pos    0       http://www.imdb.com/title/tt0453418       9  Bromwell High is a cartoon comedy. It ran at t...
                  1       http://www.imdb.com/title/tt0210075       7  If you like adult comedy cartoons, like South ...
                  2       http://www.imdb.com/title/tt0085688       9  Bromwell High is nothing short of brilliant. E...
                  3       http://www.imdb.com/title/tt0033022      10  "All the world's a stage and its people actors...
                  4       http://www.imdb.com/title/tt0043137       8  FUTZ is the only show preserved from the exper...
    """
    dfs = {}
    for subdirs in tqdm(list(product(*subdirectories))):
        urlspath = os.path.join(dirpath, subdirs[0], 'urls_{}.txt'.format(subdirs[1]))
        if not os.path.isfile(urlspath):
            if subdirs != ('test', 'unsup'):  # test/ dir doesn't usually have an unsup subdirectory
                logger.warning('Unable to find expected IMDB review list of URLs: {}'.format(urlspath))
            continue
        df = pd.read_csv(urlspath, header=None, names=['url'])
        # df.index.name = 'id'
        df['url'] = series_strip(df.url, endswith='/usercomments')

        textsdir = os.path.join(dirpath, subdirs[0], subdirs[1])
        if not os.path.isdir(textsdir):
            logger.warning('Unable to find expected IMDB review text subdirectory: {}'.format(textsdir))
            continue
        filenames = [fn for fn in os.listdir(textsdir) if fn.lower().endswith('.txt')]
        df['index0'] = subdirs[0]  # TODO: column names more generic so will work on other datasets
        df['index1'] = subdirs[1]
        df['index2'] = np.array([int(fn[:-4].split('_')[0]) for fn in filenames])
        df['rating'] = np.array([int(fn[:-4].split('_')[1]) for fn in filenames])
        texts = []
        for fn in filenames:
            with open(os.path.join(textsdir, fn)) as fin:
                texts.append(fin.read())
        df['text'] = np.array(texts)
        del texts
        df.set_index('index0 index1 index2'.split(), inplace=True)
        df.sort_index(inplace=True)
        dfs[subdirs] = df
    return pd.concat(dfs.values())


def glove_df(filepath):
    """ https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python#45894001 """
    pass


def load_glove_format(filepath):
    """ https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python#45894001 """
    # glove_input_file = os.path.join(BIGDATA_PATH, filepath)
    word2vec_output_file = os.path.join(BIGDATA_PATH, filepath.split(os.path.sep)[-1][:-4] + '.w2v.txt')
    if not os.path.isfile(word2vec_output_file):  # TODO: also check file size
        glove2word2vec(glove_input_file=filepath, word2vec_output_file=word2vec_output_file)
    return KeyedVectors.load_word2vec_format(word2vec_output_file)


def load_anki_df(language='deu'):
    """ Load into a DataFrame statements in one language along with their translation into English

    >>> get_data('zsm').head()
                    eng                                zsm
    0      Are you new?                         Awak baru?
    1      I'm at home.       Saya sedang berada di rumah.
    2   I have no clue.     Saya tiada pembayang langsung.
    3   I'm not pretty.                   Saya tak cantik.
    4  I had to resign.  Saya terpaksa meletakkan jawatan.
    """
    if os.path.isfile(language):
        filepath = language
        lang = re.search('[a-z]{3}-eng/', filepath).group()[:3].lower()
    else:
        lang = (language or 'deu').lower()[:3]
        filepath = os.path.join(BIGDATA_PATH, '{}-eng'.format(lang), '{}.txt'.format(lang))
    df = pd.read_table(filepath, skiprows=1, header=None)
    df.columns = ['eng', lang]
    return df


BIG_URLS = {
    'w2v': (
        'https://www.dropbox.com/s/965dir4dje0hfi4/GoogleNews-vectors-negative300.bin.gz?dl=1',
        1647046227,
        'GoogleNews-vectors-negative300.bin.gz',
        KeyedVectors.load_word2vec_format,
        {'binary': True},
    ),
    'glove_twitter': (
        'https://nlp.stanford.edu/data/glove.twitter.27B.zip',
        1000000000,  # FIXME: make sure size check is `>=`
    ),
    'glove': (
        'https://nlp.stanford.edu/data/glove.6B.zip',
        862182613,
        os.path.join('glove.6B', 'glove.6B.50d.txt'),
        load_glove_format,
    ),
    'glove_large': (
        'https://nlp.stanford.edu/data/glove.840B.300d.zip',
        1000000000,
    ),
    'glove_medium': (
        'https://nlp.stanford.edu/data/glove.42B.300d.zip',
        1000000000,
    ),
    'slang': (
        'https://www.dropbox.com/s/43c22018fbfzypd/slang.csv.gz?dl=1',
        117633024,
    ),
    'tweets': (
        'https://www.dropbox.com/s/5gpb43c494mc8p0/tweets.csv.gz?dl=1',
        311725313,
    ),
    'lsa_tweets': (
        'https://www.dropbox.com/s/rpjt0d060t4n1mr/lsa_tweets_5589798_2003588x200.tar.gz?dl=1',
        3112841563,
    ),
    'lsa_tweets_pickle': (
        'https://www.dropbox.com/s/7k0nvl2dx3hsbqp/lsa_tweets_5589798_2003588x200.pkl.projection.u.npy?dl=1',
        2990000000,
    ),
    'ubuntu_dialog_1500k': (
        'https://www.dropbox.com/s/krvi79fbsryytc2/ubuntu_dialog_1500k.csv.gz?dl=1',
        296098788,
    ),
    'ubuntu_dialog_test': (
        'https://www.dropbox.com/s/47mqbx0vgynvnnj/ubuntu_dialog_test.csv.gz?dl=1',
        31273,
    ),
    'imdb': (
        'https://www.dropbox.com/s/yviic64qv84x73j/aclImdb_v1.tar.gz?dl=1',
        84125825,
        'aclImdb',  # directory for extractall
        load_imdb_df,  # postprocessor to combine text files into a single DataFrame
    ),
    'imdb_test': (
        'https://www.dropbox.com/s/cpgrf3udzkbmvuu/aclImdb_test.tar.gz?dl=1',
        10858,
        'aclImdb_test',  # directory for extractall
        load_imdb_df,
    ),
    'alice': (
        # 'https://www.dropbox.com/s/py952zad3mntyvp/aiml-en-us-foundation-alice.v1-9.zip?dl=1',
        'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/' \
        'aiml-en-us-foundation-alice/aiml-en-us-foundation-alice.v1-9.zip',
        8249482,
    ),
    # BRFSS annual mental health survey
    'cdc': (
        'https://www.cdc.gov/brfss/annual_data/2016/files/LLCP2016ASC.zip',
        52284490,
    ),
}
for yr in range(2011, 2017):
    BIG_URLS['cdc' + str(yr)[-2:]] = ('https://www.cdc.gov/brfss/annual_data/{yr}/files/LLCP{yr}ASC.zip'.format(yr=yr), None)
BIG_URLS['word2vec'] = BIG_URLS['w2v']
BIG_URLS['glove_small'] = BIG_URLS['glove']
BIG_URLS['ubuntu'] = BIG_URLS['ubuntu_dialog'] = BIG_URLS['ubuntu_dialog_1500k']

ANKI_LANGUAGES = 'afr arq ara aze eus bel ben ber bul yue cat cbk cmn chv hrv ces dan nld est fin fra glg kat ' \
                 'deu ell heb hin hun isl ind ita jpn kha khm kor lvs lit nds mkd zsm mal mri mar max nob pes ' \
                 'pol por ron rus srp slk slv spa swe tgl tam tat tha tur ukr urd uig vie'.split()
LANG2ANKI = dict((lang[:2], lang) for lang in ANKI_LANGUAGES)
"""
>>> len(ANKI_LANGUAGES) - len(LANG2ANKI)
9
"""
for lang in ANKI_LANGUAGES:
    BIG_URLS[lang] = ('http://www.manythings.org/anki/{}-eng.zip'.format(lang), 1000, '{}-eng'.format(lang), load_anki_df)

"""
Google N-Gram Viewer meta data is from:
* [GOOGLE_NGRAM files](https://storage.googleapis.com/books/ngrams/books/datasetsv2.html)
* [GOOGLE_NGRAM data format](https://books.google.com/ngrams/info)
"""
GOOGLE_NGRAM_URL = 'http://storage.googleapis.com/books/ngrams/books/'
GOOGLE_NGRAM_NAMES = '0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o other p pos punctuation q r s t u v w x y z'.split()
GOOGLE_NGRAM_FILE = 'googlebooks-eng-all-1gram-20120701-{}.gz'

for name in GOOGLE_NGRAM_NAMES: 
    BIG_URLS['1gram_{}'.format(name)] = (GOOGLE_NGRAM_URL + GOOGLE_NGRAM_FILE.format(name),
                                         1000, GOOGLE_NGRAM_FILE.format(name),
                                         pd.read_table,
                                         {'sep': '\t', 'header': None, 'names': 'term_pos year term_freq book_freq'.split()})
try:
    BIGDATA_INFO = pd.read_csv(BIGDATA_INFO_FILE, header=0)
    logger.warning('Found BIGDATA index in {default} so it will overwrite nlpia.loaders.BIGDATA_URLS !!!'.format(
        default=BIGDATA_INFO_FILE))
except (IOError, pd.errors.EmptyDataError):
    BIGDATA_INFO = pd.DataFrame(columns='name url file_size'.split())
    logger.info('No BIGDATA index found in {default} so copy {latest} to {default} if you want to "freeze" it.'.format(
        default=BIGDATA_INFO_FILE, latest=BIGDATA_INFO_LATEST))
BIG_URLS.update(dict(zip(BIGDATA_INFO.name, zip(BIGDATA_INFO.url, BIGDATA_INFO.file_size))))
BIGDATA_INFO = pd.DataFrame(list(
    zip(BIG_URLS.keys(), list(zip(*BIG_URLS.values()))[0], list(zip(*BIG_URLS.values()))[1])),
    columns='name url file_size'.split())
BIGDATA_INFO.to_csv(BIGDATA_INFO_LATEST)


# FIXME: consolidate with DATA_INFO or BIG_URLS
DATA_NAMES = {
    'pointcloud': os.path.join(DATA_PATH, 'pointcloud.csv.gz'),
    'hutto_tweets0': os.path.join(DATA_PATH, 'hutto_ICWSM_2014/tweets_GroundTruth.csv.gz'),
    'hutto_tweets': os.path.join(DATA_PATH, 'hutto_ICWSM_2014/tweets_GroundTruth.csv'),
    'hutto_nyt': os.path.join(DATA_PATH, 'hutto_ICWSM_2014/nytEditorialSnippets_GroundTruth.csv.gz'),
    'hutto_movies': os.path.join(DATA_PATH, 'hutto_ICWSM_2014/movieReviewSnippets_GroundTruth.csv.gz'),
    'hutto_products': os.path.join(DATA_PATH, 'hutto_ICWSM_2014/amazonReviewSnippets_GroundTruth.csv.gz'),
}

# FIXME: put these in BIG_URLS, and test/use them with get_data()
DDL_DS_QUESTIONS_URL = 'http://minimum-entropy.districtdatalabs.com/api/questions/?format=json'
DDL_DS_ANSWERS_URL = 'http://minimum-entropy.districtdatalabs.com/api/answers/?format=json'


TEXTS = ['kite_text.txt', 'kite_history.txt']
CSVS = ['mavis-batey-greetings.csv', 'sms-spam.csv']

DATA_INFO = pd.read_csv(DATA_INFO_FILE, header=0)


def rename_file(source, dest):
    """ Rename (mv) file(s) from source to dest 

    >>> from tempfile import mkdtemp
    >>> tmpdir = mkdtemp(suffix='doctest_rename_file', prefix='tmp')
    >>> fout = open(os.path.join(tmpdir, 'fake_data.bin.gz'), 'w')
    >>> fout.write('fake nlpia.loaders.rename_file')
    30
    >>> fout.close()
    >>> dest = rename_file(os.path.join(tmpdir, 'fake_data.bin.gz'), os.path.join(tmpdir, 'Fake_Data.bin.gz'))
    >>> os.path.isfile(os.path.join(tmpdir, 'Fake_Data.bin.gz'))
    True
    """
    logger.debug('nlpia.loaders.rename_file(source={}, dest={})'.format(source, dest))
    if not isinstance(source, str):
        dest = [dest] if isinstance(dest, str) else dest
        return [rename_file(s, d) for (s, d) in zip_longest(source, dest, fillvalue=[source, dest][int(len(source) > len(dest))])]
    logger.debug('nlpia.loaders.os.rename(source={}, dest={})'.format(source, dest))
    if source == dest:
        return dest
    os.rename(source, dest)
    return dest


def normalize_ext_rename(filepath):
    """ normalize file ext like '.tgz' -> '.tar.gz' and '300d.txt' -> '300d.glove.txt' and rename the file

    >>> pth = os.path.join(DATA_PATH, 'sms_slang_dict.txt')
    >>> pth == normalize_ext_rename(pth)
    True
    """
    logger.debug('normalize_ext.filepath=' + str(filepath))
    new_file_path = normalize_ext(filepath)
    logger.debug('download_unzip.new_filepaths=' + str(new_file_path))
    # FIXME: fails when name is a url filename
    filepath = rename_file(filepath, new_file_path)
    logger.debug('download_unzip.filepath=' + str(filepath))
    return filepath


def untar(fname, verbose=True):
    """ Uunzip and untar a tar.gz file into a subdir of the BIGDATA_PATH directory """
    if fname.lower().endswith(".tar.gz"):
        dirpath = os.path.join(BIGDATA_PATH, os.path.basename(fname)[:-7])
        if os.path.isdir(dirpath):
            return dirpath
        with tarfile.open(fname) as tf:
            members = tf.getmembers()
            for member in tqdm(members, total=len(members)):
                tf.extract(member, path=BIGDATA_PATH)
        dirpath = os.path.join(BIGDATA_PATH, members[0].name)
        if os.path.isdir(dirpath):
            return dirpath
    else:
        logger.warning("Not a tar.gz file: {}".format(fname))


def series_rstrip(series, endswith='/usercomments', ignorecase=True):
    """ Strip a suffix str (`endswith` str) from a `df` columns or pd.Series of type str """
    return series_strip(series, startswith=None, endswith=endswith, startsorendswith=None, ignorecase=ignorecase)


def series_lstrip(series, startswith='http://', ignorecase=True):
    """ Strip a suffix str (`endswith` str) from a `df` columns or pd.Series of type str """
    return series_strip(series, startswith=startswith, endswith=None, startsorendswith=None, ignorecase=ignorecase)


def series_strip(series, startswith=None, endswith=None, startsorendswith=None, ignorecase=True):
    """ Strip a suffix/prefix str (`endswith`/`startswith` str) from a `df` columns or pd.Series of type str """
    if ignorecase:
        mask = series.str.lower()
        endswith = endswith.lower()
    else:
        mask = series
    if not (startsorendswith or endswith or startswith):
        logger.warning('In series_strip(): You must specify endswith, startswith, or startsorendswith string arguments.')
        return series
    if startsorendswith:
        startswith = endswith = startsorendswith
    if endswith:
        mask = mask.str.endswith(endswith)
        series[mask] = series[mask].str[:-len(endswith)]
    if startswith:
        mask = mask.str.endswith(startswith)
        series[mask] = series[mask].str[len(startswith):]
    return series


def endswith_strip(s, endswith='.txt', ignorecase=True):
    """ Strip a suffix from the end of a string

    >>> endswith_strip('http://TotalGood.com', '.COM')
    'http://TotalGood'
    >>> endswith_strip('http://TotalGood.com', endswith='.COM', ignorecase=False)
    'http://TotalGood.com'
    """
    if ignorecase:
        if s.lower().endswith(endswith.lower()):
            return s[:-len(endswith)]
    else:
        if s.endswith(endswith):
            return s[:-len(endswith)]
    return s


def startswith_strip(s, startswith='http://', ignorecase=True):
    """ Strip a prefix from the beginning of a string

    >>> startswith_strip('HTtp://TotalGood.com', 'HTTP://')
    'TotalGood.com'
    >>> startswith_strip('HTtp://TotalGood.com', startswith='HTTP://', ignorecase=False)
    'HTtp://TotalGood.com'
    """
    if ignorecase:
        if s.lower().startswith(startswith.lower()):
            return s[len(startswith):]
    else:
        if s.endswith(startswith):
            return s[len(startswith):]
    return s


def combine_dfs(dfs, index_col='index0 index1 index2'.split()):
    if isinstance(dfs, 'dict'):
        dfs = list(dfs.values())


for filename in TEXTS:
    with open(os.path.join(DATA_PATH, filename)) as fin:
        locals()[filename.split('.')[0]] = fin.read()
del fin


def looks_like_index(series, index_names=('Unnamed: 0', 'pk', 'index', '')):
    """ Tries to infer if the Series (usually leftmost column) should be the index_col

    >>> looks_like_index(pd.Series(np.arange(100)))
    True
    """
    if series.name in index_names:
        return True
    if (series == series.index.values).all():
        return True
    if (series == np.arange(len(series))).all():
        return True
    if (
        (series.index == np.arange(len(series))).all() and
        str(series.dtype).startswith('int') and
        (series.count() == len(series))
    ):
        return True
    return False


def get_longest_table(url='https://www.openoffice.org/dev_docs/source/file_extensions.html', header=0):
    """ Retrieve the HTML tables from a URL and return the longest DataFrame found 

    >>> get_longest_table('https://en.wikipedia.org/wiki/List_of_sovereign_states').columns
    Index(['Common and formal names', 'Membership within the UN System[a]',
       'Sovereignty dispute[b]',
       'Further information on status and recognition of sovereignty[d]'],
      dtype='object')
    """
    dfs = pd.read_html(url, header=header)
    return longest_table(dfs)


def get_leet_map():
    """ Retrieve mapping from English letters to l33t like E => 3 or A => /\ or /-\ or @ """
    df = get_longest_table(
        'https://sites.google.com/site/inhainternetlanguage/different-internet-languages/l33t/list-of-l33ts', header=None)
    df = df.drop(index=0).iloc[:, :2]
    df.columns = ['eng', 'l33t']
    df['l33t'] = df['l33t'].str.split(',')
    table = []
    for i, row in df.iterrows():
        for s in row['l33t']:
            table.append((row['eng'].strip(), s.strip()))
    table = pd.DataFrame(table, columns=df.columns)
    leet_path = os.path.join(DATA_PATH, 'l33t.csv')
    logger.info('Saving l33t dictionary (character mapping) to {}'.format(leet_path))
    table.to_csv(leet_path)
    return table


def get_netspeak_map():
    """ Retrieve mapping from chat/text abbreviations and acronyms like LMK => Let Me Know """
    dfs = pd.read_html('https://www.webopedia.com/quick_ref/textmessageabbreviations.asp')
    df = dfs[0].drop(index=0)
    df.columns = ['abbrev', 'definition']
    csv_path = os.path.join(DATA_PATH, 'netspeak.csv')
    logger.info('Saving netspeak dictionary (word mapping) to {}'.format(csv_path))
    df.to_csv(csv_path)
    return df


# more nontabular lists at 'https://simple.wikipedia.org/wiki/Leet


def longest_table(dfs):
    """ Return this single longest DataFrame that among an array/list/tuple of DataFrames 

    Useful for automagically finding the DataFrame you want when using pd.read_html() on a Wikipedia page.
    """
    sorted_indices = sorted((len(df if hasattr(df, '__len__') else []), i) for i, df in enumerate(dfs))
    return dfs[sorted_indices[-1][1]]


def get_filename_extensions(url='https://www.webopedia.com/quick_ref/fileextensionsfull.asp'):
    """ Load a DataFrame of filename extensions from the indicated url

    >>> df = get_filename_extensions('https://www.openoffice.org/dev_docs/source/file_extensions.html')
    >>> df.head(2)
        ext                      description
    0    .a        UNIX static library file.
    1  .asm  Non-UNIX assembler source file.
    """
    df = get_longest_table(url)
    columns = list(df.columns)
    columns[0] = 'ext'
    columns[1] = 'description'
    if len(columns) > 2:
        columns[2] = 'details'
    df.columns = columns
    return df


def read_csv(*args, **kwargs):
    """Like pandas.read_csv, only little smarter: check left column to see if it should be the index_col

    >>> read_csv(os.path.join(DATA_PATH, 'mavis-batey-greetings.csv')).head()
                                                    sentence  is_greeting
    0     It was a strange little outfit in the cottage.            0
    1  Organisation is not a word you would associate...            0
    2  When I arrived, he said: "Oh, hello, we're bre...            0
    3                                       That was it.            0
    4                I was never really told what to do.            0
    """
    kwargs.update({'low_memory': False})
    if isinstance(args[0], pd.DataFrame):
        df = args[0]
    else:
        logger.info('Reading CSV with `read_csv(*{}, **{})`...'.format(args, kwargs))
        df = pd.read_csv(*args, **kwargs)
    if looks_like_index(df[df.columns[0]]):
        df = df.set_index(df.columns[0], drop=True)
        if df.index.name in ('Unnamed: 0', ''):
            df.index.name = None
    if ((str(df.index.values.dtype).startswith('int') and (df.index.values > 1e9 * 3600 * 24 * 366 * 10).any()) or
            (str(df.index.values.dtype) == 'object')):
        try:
            df.index = pd.to_datetime(df.index)
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
            logger.info('Unable to coerce DataFrame.index into a datetime using pd.to_datetime([{},...])'.format(
                df.index.values[0]))
    return df


def wc(f, verbose=False):
    """ Count lines in a text file 

    References:
        https://stackoverflow.com/q/845058/623735

    >>> with open(os.path.join(DATA_PATH, 'dictionary_fda_drug_names.txt')) as fin:
    ...     print(wc(fin) == wc(fin) == 7037 == wc(fin.name))
    True
    >>> wc(fin.name)
    7037
    """
    if not hasattr(f, 'readlines'):
        with open(f, 'r') as fin:
            return wc(fin)
    else:
        tqdm_prog = tqdm if verbose else no_tqdm
        for i, line in tqdm_prog(enumerate(f)):
            pass
        f.seek(0)
        return i + 1


def read_txt(fin, nrows=None, verbose=True):
    lines = []
    if isinstance(fin, str):
        fin = open(fin)
    tqdm_prog = tqdm if verbose else no_tqdm
    with fin:
        for line in tqdm_prog(fin, total=wc(fin)):
            lines += [line.rstrip('\n').rstrip('\r')]
            if nrows is not None and len(lines) >= nrows:
                break
        lines = np.array(lines)
        if all('\t' in line for line in lines):
            num_tabs = [sum([1 for c in line if c == '\t']) for line in lines]
            if all(i == num_tabs[0] for i in num_tabs):
                fin.seek(0)
                return read_csv(fin, sep='\t', header=None)
    return lines


for filename in CSVS:
    locals()['df_' + filename.split('.')[0].replace('-', '_')] = read_csv(
        os.path.join(DATA_PATH, filename))


harry_docs = ["The faster Harry got to the store, the faster and faster Harry would get home.",
              "Harry is hairy and faster than Jill.",
              "Jill is not as hairy as Harry."]


def no_tqdm(it, total=1, **kwargs):
    return it


def dropbox_basename(url):
    filename = os.path.basename(url)
    match = re.findall(r'\?dl=[0-9]$', filename)
    if match:
        return filename[:-len(match[0])]
    return filename


def expand_filepath(filepath):
    """ Make sure filepath doesn't include unexpanded shortcuts like ~ and . 

    See also: pugnlp.futil.expand_path

    >>> len(expand_filepath('~')) > 3
    True 
    """
    return os.path.abspath(os.path.expandvars(os.path.expanduser(filepath)))


def normalize_ext(filepath):
    """ Convert file extension(s) to normalized form, e.g. '.tgz' -> '.tar.gz'

    Normalized extensions are ordered in reverse order of how they should be processed.
    Also extensions are ordered in order of decreasing specificity/detail.
    e.g. zip last, then txt/bin, then model type, then model dimensionality

    .TGZ => .tar.gz
    .ZIP => .zip
    .tgz => .tar.gz
    .bin.gz => .w2v.bin.gz
    .6B.zip => .6B.glove.txt.zip
    .27B.zip => .27B.glove.txt.zip
    .42B.300d.zip => .42B.300d.glove.txt.zip
    .840B.300d.zip => .840B.300d.glove.txt.zip

    TODO: use regexes to be more general (deal with .300D and .42B extensions)

    >>> normalize_ext('glove.6B.zip').endswith('glove.6b.glove.txt.zip')
    True
    >>> normalize_ext('glove.twitter.27B.zip').endswith('glove.twitter.27b.glove.txt.zip')
    True
    >>> normalize_ext('glove.42B.300d.zip').endswith('glove.42b.300d.glove.txt.zip')
    True
    >>> normalize_ext('glove.840B.300d.zip').endswith('glove.840b.300d.glove.txt.zip')
    True
    """
    mapping = tuple(reversed((
        ('.tgz', '.tar.gz'),
        ('.bin.gz', '.w2v.bin.gz'),
        ('.6B.zip', '.6b.glove.txt.zip'),
        ('.27B.zip', '.27b.glove.txt.zip'),
        ('.42B.300d.zip', '.42b.300d.glove.txt.zip'),
        ('.840B.300d.zip', '.840b.300d.glove.txt.zip'),
    )))
    if not isinstance(filepath, str):
        return [normalize_ext(fp) for fp in filepath]
    if '~' in filepath or os.path.sep in filepath:
        filepath = expand_filepath(filepath)
    fplower = filepath.lower()
    for ext, newext in mapping:
        ext = ext.lower()
        if fplower.endswith(ext) and not fplower.endswith(newext):
            filepath = filepath[:-len(ext)] + newext
    return filepath


def normalize_filepath(filepath):
    r""" Lowercase the filename and ext, expanding extensions like .tgz to .tar.gz. 

    >>> normalize_filepath('/Hello_World.txt\n')
    'hello_world.txt'
    >>> normalize_filepath('NLPIA/src/nlpia/bigdata/Goog New 300Dneg\f.bIn\n.GZ')
    'NLPIA/src/nlpia/bigdata/goog new 300dneg.w2v.bin.gz'
    """
    filename = os.path.basename(filepath)
    dirpath = filepath[:-len(filename)]
    cre_controlspace = re.compile(r'[\t\r\n\f]+')
    new_filename = cre_controlspace.sub('', filename)
    if not new_filename == filename:
        logger.warning('Stripping whitespace from filename: {} => {}'.format(
            repr(filename), repr(new_filename)))
        filename = new_filename
    filename = filename.lower()
    filename = normalize_ext(filename)
    if dirpath:
        dirpath = dirpath[:-1]  # get rid of the trailing os.path.sep
        return os.path.join(dirpath, filename)
    return filename


def migrate_big_urls(big_urls=BIG_URLS, inplace=True):
    r""" Migrate the big_urls table schema/structure from a dict of lists to a dict of dicts

    >>> big_urls = {'x': (1, 2, 3, "4x"), 'y': ("yme", "cause")}
    >>> inplace = migrate_big_urls(big_urls=big_urls)
    >>> inplace
    {'x': {0: 1, 1: 2, 2: 3, 3: '4x'}, 'y': {0: 'yme', 1: 'cause'}}
    >>> inplace is big_urls
    True
    >>> big_urls = {'x': [1, 2, 3, "4x"], 'y': ["yme", "cause"]}
    >>> copied = migrate_big_urls(big_urls=big_urls, inplace=False)
    >>> copied
    {'x': {0: 1, 1: 2, 2: 3, 3: '4x'}, 'y': {0: 'yme', 1: 'cause'}}
    >>> copied is big_urls
    False
    >>> copied['x'] is big_urls['x']
    False
    >>> 1 is copied['x'][0] is big_urls['x'][0]
    True
    """
    if not inplace:
        big_urls = deepcopy(big_urls)
    for name, meta in big_urls.items():
        big_urls[name] = dict(zip(range(len(meta)), meta))
        big_urls[name] = dict(zip(range(len(meta)), meta))
        # big_urls[name]['filenames'] = [normalize_ext(big_urls)]
    return big_urls


BIG_URLS = migrate_big_urls(BIG_URLS)


def normalize_glove(filepath):
    r""" https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python#45894001 """
    # FIXME
    filepath = expand_filepath(filepath)
    raise NotImplementedError()


def unzip(filepath, verbose=True):
    r""" Unzip GloVE models and convert to word2vec binary models (gensim.KeyedVectors) 

    The only kinds of files that are returned are "*.asc" and "*.txt" and only after renaming.
    """
    filepath = expand_filepath(filepath)
    filename = os.path.basename(filepath)
    tqdm_prog = tqdm if verbose else no_tqdm
    z = ZipFile(filepath)

    unzip_dir = filename.split('.')[0] if filename.split('.')[0] else os.path.splitext(filename)[0]
    unzip_dir = os.path.join(BIGDATA_PATH, unzip_dir)
    if not os.path.isdir(unzip_dir) or not len(os.listdir(unzip_dir)) == len(z.filelist):
        z.extractall(path=unzip_dir)

    logger.info('unzip_dir contains: {}'.format(os.listdir(unzip_dir)))
    # for f in os.listdir(unzip_dir):
    #     if f.lower().endswith('about.txt'):
    #         os.remove(os.path.join(unzip_dir, f))
    for f in tqdm_prog(os.listdir(unzip_dir)):
        if f[-1] in ' \t\r\n\f':
            bad_path = os.path.join(unzip_dir, f)
            logger.warning('Stripping whitespace from end of filename: {} -> {}'.format(
                repr(bad_path), repr(bad_path.rstrip())))
            shutil.move(bad_path, bad_path.rstrip())
            # rename_file(source=bad_path, dest=bad_path.rstrip())
    anki_paths = [os.path.join(unzip_dir, f) for f in os.listdir(unzip_dir)
                  if f.lower()[:3] in ANKI_LANGUAGES and f.lower()[3:] == '.txt']
    logger.info('anki_paths: {}'.format(anki_paths))

    w2v_paths = [os.path.join(BIGDATA_PATH, f[:-4] + '.w2v.txt') for f in os.listdir(unzip_dir)
                 if f.lower().endswith('.txt') and 'glove' in f.lower()]
    for f, word2vec_output_file in zip(os.listdir(unzip_dir), w2v_paths):
        glove_input_file = os.path.join(unzip_dir, f)
        logger.info('Attempting to converting GloVE format to Word2vec: {} -> {}'.format(
            repr(glove_input_file), repr(word2vec_output_file)))
        try:
            glove2word2vec(glove_input_file=glove_input_file, word2vec_output_file=word2vec_output_file)
        except:
            logger.info('Failed to convert GloVE format to Word2vec: {} -> {}'.format(
                repr(glove_input_file), repr(word2vec_output_file)))

    txt_paths = [os.path.join(BIGDATA_PATH, f.lower()[:-4] + '.txt') for f in os.listdir(unzip_dir) if f.lower().endswith('.asc')]
    for f, txt_file in zip(os.listdir(unzip_dir), txt_paths):
        if f.lower().endswith('.asc'):
            input_file = os.path.join(unzip_dir, f)
            logger.info('Renaming .asc file to .txt: {} -> {}'.format(
                repr(input_file), repr(txt_file)))
            shutil.move(input_file, txt_file)

    return anki_paths + txt_paths + w2v_paths


def create_big_url(name):
    """ If name looks like a url, with an http, add an entry for it in BIG_URLS """
    # BIG side effect
    global BIG_URLS
    filemeta = get_url_filemeta(name)
    if not filemeta:
        return None
    filename = filemeta['filename']
    remote_size = filemeta['remote_size']
    url = filemeta['url']
    name = filename.split('.')
    name = (name[0] if name[0] not in ('', '.') else name[1]).replace(' ', '-')
    name = name.lower().strip()
    BIG_URLS[name] = (url, int(remote_size or -1), filename)
    return name


def try_parse_url(url):
    """ User urlparse to try to parse URL returning None on exception """
    if len(url.strip()) < 4:
        logger.error('Invalid URL: {}'.format(url))
        return None
    try:
        parsed_url = urlparse(url)
    except ValueError:
        logger.error('Invalid URL: {}'.format(url))
        return None
    if parsed_url.scheme:
        return parsed_url
    try:
        parsed_url = urlparse('http://' + parsed_url.geturl())
    except ValueError:
        logger.error('Invalid URL: {}'.format(url))
        return None
    if not parsed_url.scheme:
        logger.error('Unable to guess a scheme for URL: {}'.format(url))
        return None
    return parsed_url


def get_ftp_filemeta(parsed_url, username='anonymous', password='nlpia@totalgood.com'):
    """ FIXME: Get file size, hostname, path metadata from FTP server using parsed_url (urlparse)"""
    return dict(
        url=parsed_url.geturl(), hostname=parsed_url.hostname, path=parsed_url.path,
        username=(parsed_url.username or username),
        remote_size=-1,
        filename=os.path.basename(parsed_url.path))
    ftp = ftplib.FTP(parsed_url.hostname) 
    ftp.login(username, password) 
    ftp.cwd(parsed_url.path)
    ftp.retrbinary("RETR " + filename, open(filename, 'wb').write)
    ftp.quit()


def get_url_filemeta(url):
    """ Request HTML for the page at the URL indicated and return the url, filename, and remote size

    TODO: just add remote_size and basename and filename attributes to the urlparse object 
          instead of returning a dict

    >>> sorted(get_url_filemeta('mozilla.com').items())
    [('filename', ''),
     ('hostname', 'mozilla.com'),
     ('path', ''),
     ('remote_size', -1),
     ('url', 'http://mozilla.com'),
     ('username', None)]
    >>> sorted(get_url_filemeta('https://duckduckgo.com/about?q=nlp').items())
    [('filename', 'about'),
     ('hostname', 'duckduckgo.com'),
     ('path', '/about'),
     ('remote_size', -1),
     ('url', 'https://duckduckgo.com/about?q=nlp'),
     ('username', None)]
    >>> 1000 <= int(get_url_filemeta('en.wikipedia.org')['remote_size']) <= 200000
    True
    """ 
    parsed_url = try_parse_url(url)

    if parsed_url is None:
        return None
    if parsed_url.scheme.startswith('ftp'):
        return get_ftp_filemeta(parsed_url)

    url = parsed_url.geturl()
    try:
        r = requests.get(url, stream=True, allow_redirects=True, timeout=5)
        remote_size = r.headers.get('Content-Length', -1)
        return dict(url=url, hostname=parsed_url.hostname, path=parsed_url.path,
                    username=parsed_url.username, remote_size=remote_size,
                    filename=os.path.basename(parsed_url.path))
    except ConnectionError:
        return None
    except (InvalidURL, InvalidSchema, InvalidHeader, MissingSchema):
        return None
    return None


def get_url_title(url):
    r""" Request HTML for the page at the URL indicated and return it's <title> property

    >>> get_url_title('mozilla.com').strip()
    'Internet for people, not profit\n    — Mozilla'
    """
    parsed_url = try_parse_url(url)
    if parsed_url is None:
        return None
    try:
        r = requests.get(parsed_url.geturl(), stream=False, allow_redirects=True, timeout=5)
        tree = parse_html(r.content)
        title = tree.findtext('.//title')
        return title
    except ConnectionError:
        logging.error('Unable to connect to internet to retrieve URL {}'.format(parsed_url.geturl()))
        logging.error(format_exc())
    except (InvalidURL, InvalidSchema, InvalidHeader, MissingSchema):
        logging.warn('Unable to retrieve URL {}'.format(parsed_url.geturl()))
        logging.error(format_exc())


def download_unzip(names=None, verbose=True):
    r""" Download CSV or HTML tables listed in `names`, unzip and to DATA_PATH/`names`.csv .txt etc

    Also normalizes file name extensions (.bin.gz -> .w2v.bin.gz).
    Uses table in data_info.csv (internal DATA_INFO) to determine URL or file path from dataset name.
    Also looks

    If names or [names] is a valid URL then download it and create a name
        from the url in BIGDATA_URLS (not yet pushed to data_info.csv)
    """
    names = [names] if isinstance(names, (str, basestring)) else names
    # names = names or list(BIG_URLS.keys())  # download them all, if none specified!
    file_paths = {}
    for name in names:
        created = create_big_url(name)
        name = (created or name).lower().strip()

        if name in BIG_URLS:
            filepath = download_name(name, verbose=verbose)
            if not filepath:
                continue
            file_paths[name] = normalize_ext_rename(filepath)
            logger.debug('downloaded name={} to filepath={}'.format(name, file_paths[name]))
            fplower = file_paths[name].lower()
            if fplower.endswith('.tar.gz'):
                logger.info('Extracting {}'.format(file_paths[name]))
                file_paths[name] = untar(file_paths[name], verbose=verbose)
                logger.debug('download_untar.filepaths=' + str(file_paths))
            elif file_paths[name].lower().endswith('.zip'):
                file_paths[name] = unzip(file_paths[name], verbose=verbose)
                logger.debug('download_unzip.filepaths=' + str(file_paths))
        else:
            df = pd.read_html(DATA_INFO['url'][name], **DATA_INFO['downloader_kwargs'][name])[-1]
            df.columns = clean_columns(df.columns)
            file_paths[name] = os.path.join(DATA_PATH, name + '.csv')
            df.to_csv(file_paths[name])
        file_paths[name] = normalize_ext_rename(file_paths[name])
    return file_paths


download = download_unzip


def download_file(url, data_path=BIGDATA_PATH, filename=None, size=None, chunk_size=4096, verbose=True):
    """Uses stream=True and a reasonable chunk size to be able to download large (GB) files over https

    Downloading this small file takes 1.5 sec. All subsequent "downloads" takes .6 sec to verify path and size.
    >>> import time
    >>> meta = BIG_URLS['ubuntu_dialog_test']
    >>> pathend = os.path.join(*('nlpia/src/nlpia/bigdata/ubuntu_dialog_test.csv.gz'.split('/')))
    >>> download_file(url=meta[0], verbose=False).endswith(pathend)
    True
    >>> t0 = time.time()
    >>> localpath = download_file(url=BIG_URLS['ubuntu_dialog_test'][0], verbose=False)
    >>> t1 = time.time()
    >>> localpath is None or ((0.015 < (t1 - t0) < 5.0) and localpath.endswith(pathend))
    True
    >>> t0 = time.time()
    >>> download_file(url=meta[0], size=meta[1], verbose=False).endswith(pathend)
    True
    >>> time.time() - t0 < 0.02
    True
    """ 
    if isinstance(url, (list, tuple)):
        return [
            download_file(
                s, data_path=data_path, filename=filename, size=size, chunk_size=chunk_size, verbose=verbose)
            for s in url]
    if url.endswith('dl=0'):
        url = url[:-1] + '1'  # noninteractive Dropbox download
    remote_size = size

    # figure out what filename to expect after download and how big it should be
    if filename is None:
        filename = dropbox_basename(url)
    filepath = normalize_filepath(os.path.join(data_path, filename))
    logger.info('expanded+normalized file path: {}'.format(filepath))
    tqdm_prog = tqdm if verbose else no_tqdm
    logger.info('requesting URL: {}'.format(url))

    logger.info('remote_size: {}'.format(remote_size))
    stat = path_status(filepath)
    local_size = stat.get('size', None)
    logger.info('local_size: {}'.format(local_size))

    r = None
    if not remote_size or not stat['type'] == 'file' or not local_size >= remote_size or not stat['size'] > MIN_DATA_FILE_SIZE:
        try:
            r = requests.get(url, stream=True, allow_redirects=True, timeout=5)
            remote_size = r.headers.get('Content-Length', -1)
        except ConnectionError:
            logger.error('ConnectionError for url: {} => request {}'.format(url, r))
            remote_size = -1 if remote_size is None else remote_size
        except (InvalidURL, InvalidSchema, InvalidHeader, MissingSchema) as e:
            logger.error(e)
            logger.error('HTTP Error for url: {}\n request: {}\n traceback: {}'.format(url, r, format_exc()))
    try:
        remote_size = int(remote_size)
    except ValueError:
        remote_size = -1

    # remote_size has changed so need to check it again
    # TODO: check md5 or get the right size of remote file
    if stat['type'] == 'file' and local_size >= remote_size and stat['size'] > MIN_DATA_FILE_SIZE:
        r = r.close() if r else r
        logger.info('retained: {}'.format(filepath))
        return filepath

    filedir = os.path.dirname(filepath)
    created_dir = mkdir_p(filedir)
    logger.info('data path created: {}'.format(created_dir))
    assert os.path.isdir(filedir)
    assert created_dir.endswith(filedir)
    bytes_downloaded = 0
    if r:
        logger.info('downloading to: {}'.format(filepath))
        with open(filepath, 'wb') as f:
            for chunk in tqdm_prog(r.iter_content(chunk_size=chunk_size), total=ceil(remote_size / float(chunk_size))):
                bytes_downloaded += len(chunk)
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)      
        r.close()
    else:
        logger.error('Unable to request URL: {} using request object {}'.format(url, r))
        return None

    logger.debug('nlpia.loaders.download_file: bytes={}'.format(bytes_downloaded))
    stat = path_status(filepath)
    logger.info("local file stat {}".format(stat))
    logger.debug("filepath={}: remote {}, downloaded {}".format(
        filepath, size, remote_size, bytes_downloaded))
    return filepath


def download_name(name, verbose=True, **kwargs):
    meta = BIG_URLS[name]
    size = meta[1] or -1
    url = meta[0]
    return download_file(url=url, size=size, verbose=verbose, **kwargs)
    # for filename in meta['filenames']


def read_named_csv(name, data_path=DATA_PATH, nrows=None, verbose=True):
    """ Convert a dataset in a local file (usually a CSV) into a Pandas DataFrame

    TODO: should be called read_named_dataset

    Args:
    `name` is assumed not to have an extension (like ".csv"), alternative extensions are tried automatically.file
    """
    if os.path.isfile(name):
        try:
            return read_json(name)
        except (IOError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        try:
            return read_csv(name, nrows=nrows)
        except (IOError, pd.errors.ParserError):
            pass
        try:
            return read_txt(name, nrows=nrows)
        except (IOError, UnicodeDecodeError):
            pass
    data_path = expand_filepath(data_path)
    try:
        return read_csv(os.path.join(data_path, name + '.csv.gz'), nrows=nrows)
    except IOError:
        pass
    try:
        return read_csv(os.path.join(data_path, name + '.csv'), nrows=nrows)
    except IOError:
        pass
    try:
        return read_json(os.path.join(data_path, name + '.json'))
    except IOError:
        pass
    try:
        return read_txt(os.path.join(data_path, name + '.txt'), verbose=verbose)
    except IOError:
        pass

    # FIXME: mapping from short name to uncompressed filename
    # BIGDATA files are usually not loadable into dataframes
    try:
        return KeyedVectors.load_word2vec_format(os.path.join(BIGDATA_PATH, name + '.bin.gz'), binary=True)
    except IOError:
        pass
    except ValueError:
        pass
    try:
        return read_txt(os.path.join(BIGDATA_PATH, name + '.txt'), verbose=verbose)
    except IOError:
        pass


def get_data(name='sms-spam', nrows=None):
    """ Load data from a json, csv, or txt file if it exists in the data dir.

    References:
      [cities_air_pollution_index](https://www.numbeo.com/pollution/rankings.jsp)
      [cities](http://download.geonames.org/export/dump/cities.zip)
      [cities_us](http://download.geonames.org/export/dump/cities_us.zip)

    >>> from nlpia.data.loaders import get_data
    >>> words = get_data('words_ubuntu_us')
    >>> len(words)
    99171
    >>> get_data('imdb_test').info()
    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 20 entries, (train, pos, 0) to (train, neg, 9)
    Data columns (total 3 columns):
    url       20 non-null object
    rating    20 non-null int64
    text      20 non-null object
    dtypes: int64(1), object(2)
    memory usage: 809.0+ bytes
    >>> list(words[:8])
    ['A', "A's", "AA's", "AB's", "ABM's", "AC's", "ACTH's", "AI's"]
    >>> get_data('ubuntu_dialog_test').iloc[0]
    Context      i think we could import the old comments via r...
    Utterance    basically each xfree86 upload will NOT force u...
    Name: 0, dtype: object
    """
    if name in BIG_URLS:
        logger.info('Downloading {}'.format(name))
        filepaths = download_unzip(name)
        logger.debug('nlpia.loaders.get_data.filepaths=' + str(filepaths))
        filepath = filepaths[name][0] if isinstance(filepaths[name], (list, tuple)) else filepaths[name]
        logger.debug('nlpia.loaders.get_data.filepath=' + str(filepath))
        filepathlow = filepath.lower()

        if len(BIG_URLS[name]) >= 4:
            kwargs = BIG_URLS[name][4] if len(BIG_URLS[name]) >= 5 else {}
            return BIG_URLS[name][3](filepath, **kwargs)
        if filepathlow.endswith('.w2v.txt'):
            try:
                return KeyedVectors.load_word2vec_format(filepath, binary=False)
            except (TypeError, UnicodeError):
                pass
        if filepathlow.endswith('.w2v.bin') or filepathlow.endswith('.bin.gz'):
            try:
                return KeyedVectors.load_word2vec_format(filepath, binary=True)
            except (TypeError, UnicodeError):
                pass
        if filepathlow.endswith('.gz'):
            try:
                filepath = gzip.open(filepath)
            except:
                pass
        if filepathlow.endswith('.tsv.gz') or filepathlow.endswith('.tsv'):
            try:
                return pd.read_table(filepath)
            except:
                pass
        if filepathlow.endswith('.csv.gz') or filepathlow.endswith('.csv'):
            try:
                return read_csv(filepath)
            except:
                pass
        if filepathlow.endswith('.txt') or filepathlow.endswith('.bin.gz') or filepathlow.endswith('.w2v.bin.gz'):
            try:
                return read_txt(filepath)
            except (TypeError, UnicodeError):
                pass
        return filepaths[name]
    elif name in DATASET_NAME2FILENAME:
        return read_named_csv(name, data_path=DATA_PATH, nrows=nrows)
    elif name in DATA_NAMES:
        return read_named_csv(DATA_NAMES[name], nrows=nrows)
    elif os.path.isfile(name):
        return read_named_csv(name, nrows=nrows)
    elif os.path.isfile(os.path.join(DATA_PATH, name)):
        return read_named_csv(os.path.join(DATA_PATH, name), nrows=nrows)

    msg = 'Unable to find dataset "{}"" in {} or {} (*.csv.gz, *.csv, *.json, *.zip, or *.txt)\n'.format(
        name, DATA_PATH, BIGDATA_PATH)
    msg += 'Available dataset names include:\n{}'.format('\n'.join(DATASET_NAMES))
    logger.error(msg)
    raise IOError(msg)


def multifile_dataframe(paths=['urbanslang{}of4.csv'.format(i) for i in range(1, 5)], header=0, index_col=None):
    """Like pandas.read_csv, but loads and concatenates (df.append(df)s) DataFrames together"""
    df = pd.DataFrame()
    for p in paths:
        df = df.append(read_csv(p, header=header, index_col=index_col), ignore_index=True if not index_col else False)
    if index_col and df.index.name == index_col:
        del df[index_col]
    return df


def read_json(filepath):
    filepath = expand_filepath(filepath)
    return json.load(open(filepath, 'rt'))


def get_wikidata_qnum(wikiarticle, wikisite):
    """Retrieve the Query number for a wikidata database of metadata about a particular article

    >>> print(get_wikidata_qnum(wikiarticle="Andromeda Galaxy", wikisite="enwiki"))
    Q2469
    """
    resp = requests.get('https://www.wikidata.org/w/api.php', timeout=5, params={
        'action': 'wbgetentities',
        'titles': wikiarticle,
        'sites': wikisite,
        'props': '',
        'format': 'json'
    }).json()
    return list(resp['entities'])[0]


DATASET_FILENAMES = [f['name'] for f in find_files(DATA_PATH, '.csv.gz', level=0)]
DATASET_FILENAMES += [f['name'] for f in find_files(DATA_PATH, '.csv', level=0)]
DATASET_FILENAMES += [f['name'] for f in find_files(DATA_PATH, '.json', level=0)]
DATASET_FILENAMES += [f['name'] for f in find_files(DATA_PATH, '.txt', level=0)]
DATASET_NAMES = sorted(
    [f[:-4] if f.endswith('.csv') else f for f in [os.path.splitext(f)[0] for f in DATASET_FILENAMES]])
DATASET_NAME2FILENAME = dict(zip(DATASET_NAMES, DATASET_FILENAMES))


def str2int(s):
    s = ''.join(c for c in s if c in '0123456789')
    return int(s or INT_MIN)


def clean_toxoplasmosis(url='http://www.rightdiagnosis.com/t/toxoplasmosis/stats-country.htm'):
    dfs = pd.read_html('http://www.rightdiagnosis.com/t/toxoplasmosis/stats-country.htm', header=0)
    df = dfs[0].copy()
    df.columns = normalize_column_names(df.columns)
    df = df.dropna().copy()
    df['extrapolated_prevalence'] = df['extrapolated_prevalence'].apply(str2int)
    df['population_estimated_used'] = df['population_estimated_used'].apply(str2int)
    df['frequency'] = df.extrapolated_prevalence.astype(float) / df.population_estimated_used
    return df


def normalize_column_names(df):
    r""" Clean up whitespace in column names. See better version at `pugnlp.clean_columns`

    >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['Hello World', 'not here'])
    >>> normalize_column_names(df)
    ['hello_world', 'not_here']
    """
    columns = df.columns if hasattr(df, 'columns') else df
    columns = [c.lower().replace(' ', '_') for c in columns]
    return columns


def clean_column_values(df, inplace=True):
    r""" Convert dollar value strings, numbers with commas, and percents into floating point values

    >>> df = get_data('us_gov_deficits_raw')
    >>> df2 = clean_column_values(df, inplace=False)
    >>> df2.iloc[0]
    Fiscal year                                                               10/2017-3/2018
    President's party                                                                      R
    Senate majority party                                                                  R
    House majority party                                                                   R
    Top-bracket marginal income tax rate                                                38.3
    National debt millions                                                       2.10896e+07
    National debt millions of 1983 dollars                                       8.47004e+06
    Deficit\n(millions of 1983 dollars)                                               431443
    Surplus string in 1983 dollars                                                       NaN
    Deficit string in 1983 dollars ($ = $10B)    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    Net surplus in 1983 dollars ($B)                                                    -430
    Name: 0, dtype: object
    """
    dollars_percents = re.compile(r'[%$,;\s]+')
    if not inplace:
        df = df.copy()
    for c in df.columns:
        values = None
        if df[c].dtype.char in '<U S O'.split():
            try:
                values = df[c].copy()
                values = values.fillna('')
                values = values.astype(str).str.replace(dollars_percents, '')
                # values = values.str.strip().str.replace(dollars_percents, '').str.strip()
                if values.str.len().sum() > .2 * df[c].astype(str).str.len().sum():
                    values[values.isnull()] = np.nan
                    values[values == ''] = np.nan
                    values = values.astype(float)
            except ValueError:
                values = None
            except:
                logger.error('Error on column {} with dtype {}'.format(c, df[c].dtype))
                raise

        if values is not None:
            if values.isnull().sum() < .6 * len(values) and values.any():
                df[c] = values
    return df


def load_geonames(filepath='http://download.geonames.org/export/dump/cities1000.zip'):
    """Clean the table of city metadata from download.geoname.org/export/dump/{filename}

    Reference:
      http://download.geonames.org/export/dump/readme.txt

    'cities1000.txt' and 'allCountries.txt' have the following tab-separated fields:

    0  geonameid         : integer id of record in geonames database
    1  name              : name of geographical point (utf8) varchar(200)
    2  asciiname         : name of geographical point in plain ascii characters, varchar(200)
    3  alternatenames    : alternatenames, comma separated, ascii names automatically transliterated,
                           convenience attribute from alternatename table, varchar(10000)
    4  latitude          : latitude in decimal degrees (wgs84)
    5  longitude         : longitude in decimal degrees (wgs84)
    6  feature class     : see http://www.geonames.org/export/codes.html, char(1)
    7  feature code      : see http://www.geonames.org/export/codes.html, varchar(10)
    8  country code      : ISO-3166 2-letter country code, 2 characters
    9  cc2               : alternate country codes, comma separated, ISO-3166 2-letter country code, 200 characters
    10 admin1 code       : fipscode (subject to change to iso code), see exceptions below,
                           see file admin1Codes.txt for display names of this code; varchar(20)
    11 admin2 code       : code for the second administrative division, a county in the US,
                           see file admin2Codes.txt; varchar(80)
    12 admin3 code       : code for third level administrative division, varchar(20)
    13 admin4 code       : code for fourth level administrative division, varchar(20)
    14 population        : bigint (8 byte int)
    15 elevation         : in meters, integer
    16 dem               : digital elevation model, srtm3 or gtopo30, average elevation of
                           (3''x3''ca 90mx90m) or 30''x30''(ca 900mx900m) area in meters, integer.
                           srtm processed by cgiar/ciat.
    17 timezone          : the iana timezone id (see file timeZone.txt) varchar(40)
    18 modification date : date of last modification in yyyy-MM-dd format
    """
    columns = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 'feature class',
               'feature code', 'country code']
    columns += ['cc2', 'admin1_code', 'admin2_code', 'admin3_code', 'admin4_code', 'population', 'elevation',
                'dem', 'timezone', 'modification date']
    columns = normalize_column_names(columns)
    df = pd.read_csv(filepath, sep='\t', index_col=None, low_memory=False, header=None)
    df.columns = columns
    return df


def load_geo_adwords(filename='AdWords API Location Criteria 2017-06-26.csv.gz'):
    """ WARN: Not a good source of city names. This table has many errors, even after cleaning"""
    df = pd.read_csv(filename, header=0, index_col=0, low_memory=False)
    df.columns = [c.replace(' ', '_').lower() for c in df.columns]
    canonical = pd.DataFrame([list(row) for row in df.canonical_name.str.split(',').values])

    def cleaner(row):
        cleaned = pd.np.array(
            [s for i, s in enumerate(row.values) if s not in ('Downtown', None) and (i > 3 or row[i + 1] != s)])
        if len(cleaned) == 2:
            cleaned = [cleaned[0], None, cleaned[1], None, None]
        else:
            cleaned = list(cleaned) + [None] * (5 - len(cleaned))
        if not pd.np.all(pd.np.array(row.values)[:3] == pd.np.array(cleaned)[:3]):
            logger.info('{} => {}'.format(row.values, cleaned))
        return list(cleaned)

    cleancanon = canonical.apply(cleaner, axis=1)
    cleancanon.columns = 'city region country extra extra2'.split()
    df['region'] = cleancanon.region
    df['country'] = cleancanon.country
    return df


def nlp(texts, lang='en', linesep=None, verbose=True):
    r""" Use the SpaCy parser to parse and tag natural language strings. 

    Load the SpaCy parser language model lazily and share it among all nlpia modules.
    Probably unnecessary, since SpaCy probably takes care of this with `spacy.load()`

    >>> _parse is None
    True
    >>> doc = nlp("Domo arigatto Mr. Roboto.")
    >>> doc.text
    'Domo arigatto Mr. Roboto.'
    >>> doc.ents
    (Roboto,)
    >>> docs = nlp("Hey Mr. Tangerine Man!\nPlay a song for me.\n", linesep='\n')
    >>> doc = docs[0]
    >>> [t for t in doc]
    [Hey, Mr., Tangerine, Man, !]
    >>> [tok.text for tok in doc]
    ['Hey', 'Mr.', 'Tangerine', 'Man', '!']
    >>> [(tok.text, tok.tag_) for tok in doc]
    [('Hey', 'UH'),
     ('Mr.', 'NNP'),
     ('Tangerine', 'NNP'),
     ('Man', 'NN'),
     ('!', '.')]
    >>> [(ent.text, ent.ent_id, ent.has_vector, ent.vector[:3].round(3)) for ent in doc.ents]
    [('Tangerine Man', 0, True, array([0.72 , 1.913, 2.675], dtype=float32))]
    """
    # doesn't let you load a different model anywhere else in the module
    linesep = os.linesep if linesep in ('default', True, 1, 'os') else linesep
    tqdm_prog = no_tqdm if (not verbose or (hasattr(texts, '__len__') and len(texts) < 3)) else tqdm
    global _parse
    if not _parse:
        try:
            _parse = spacy.load(lang)
        except (OSError, IOError):
            try:
                spacy.cli.download(lang)
            except URLError:
                logger.warning("Unable to download Spacy language model '{}' so nlp(text) just returns text.split()".format(lang))
    parse = _parse or str.split
    # TODO: reverse this recursion (str first then sequence) to allow for sequences of sequences of texts
    if isinstance(texts, str):
        if linesep:
            return nlp(texts.split(linesep))
        else: 
            return nlp([texts])
    if hasattr(texts, '__len__'):
        if len(texts) == 1:
            return parse(texts[0])
        elif len(texts) > 1:
            return [(parse or str.split)(text) for text in tqdm_prog(texts)]
        else:
            return None
    else: 
        # return generator if sequence of strings doesn't have __len__ which means its an iterable or generator itself
        return (parse(text) for text in tqdm_prog(texts))
    # TODO: return the same type as the input, e.g. `type(texts)(texts)`


def clean_win_tsv(filepath=os.path.join(DATA_PATH, 'Products.txt'),
                  index_col=False, sep='\t', lineterminator='\r', error_bad_lines=False, **kwargs):
    """ Load and clean tab-separated files saved on Windows OS ('\r\n') """
    df = pd.read_csv(filepath, index_col=index_col, sep=sep, lineterminator=lineterminator,
                     error_bad_lines=error_bad_lines, **kwargs)
    index_col = df.columns[0]
    original_len = len(df)
    if df[index_col].values[-1] == '\n':
        df.iloc[-1, 0] = np.nan
        original_len = len(df) - 1
    df.dropna(how='all', inplace=True)
    df[index_col] = df[index_col].str.strip().apply(lambda x: x if x else str(INT_MIN)).astype(int)
    df = df[~(df[index_col] == INT_NAN)]
    df.set_index(index_col, inplace=True)
    if len(df) != original_len:
        logger.warning(('Loaded {} rows from tsv. Original file, "{}", contained {} seemingly valid lines.' +
                        'Index column: {}').format(len(df), original_len, filepath, index_col))
    return df
