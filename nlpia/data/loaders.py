from __future__ import print_function, unicode_literals, division, absolute_import
from future import standard_library
standard_library.install_aliases()  # noqa
from builtins import *  # noqa

import os
import re
import json
import requests
from nlpia.constants import logging, DATA_PATH, BIGDATA_PATH

from tqdm import tqdm
from pugnlp.futil import path_status
import pandas as pd
import tarfile

"""Loaders and downloaders for data files and models required for the examples in NLP in Action

>>> from nlpia.data import download
>> download()  Will take hours and 8GB of storage
"""
np = pd.np
logger = logging.getLogger(__name__)

# SMALLDATA_URL = 'http://totalgood.org/static/data'
W2V_FILE = 'GoogleNews-vectors-negative300.bin.gz'
BIG_URLS = {
    'w2v': (
        'https://www.dropbox.com/s/965dir4dje0hfi4/GoogleNews-vectors-negative300.bin.gz?dl=1',
        1647046227,
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
        3112841563,  # 3112841312,
        ),
    'imdb': (
        'https://www.dropbox.com/s/yviic64qv84x73j/aclImdb_v1.tar.gz?dl=1',
        3112841563,  # 3112841312,
        ),
    }
DATA_NAMES = {
    'pointcloud': os.path.join(DATA_PATH, 'pointcloud.csv.gz')
}

DDL_DS_QUESTIONS_URL = 'http://minimum-entropy.districtdatalabs.com/api/questions/?format=json'
DDL_DS_ANSWERSS_URL = 'http://minimum-entropy.districtdatalabs.com/api/answers/?format=json'


W2V_PATH = os.path.join(BIGDATA_PATH, W2V_FILE)
TEXTS = ['kite_text.txt', 'kite_history.txt']
CSVS = ['mavis-batey-greetings.csv', 'sms-spam.csv']


def untar(fname):
    if fname.endswith("tar.gz"):
        with tarfile.open(fname) as tf:
            tf.extractall()
    else:
        print("Not a tar.gz file: {}".format(fname))


for filename in TEXTS:
    with open(os.path.join(DATA_PATH, filename)) as f:
        locals()[filename.split('.')[0]] = f.read()


def read_csv(*args, **kwargs):
    """Like pandas.read_csv, only little smarter (checks first column to see if it should be the data frame index)

    >>> read_csv('mavis-batey-greetings.csv').head()
    """
    index_names = ('Unnamed: 0', 'pk', 'index', '')
    kwargs.update({'low_memory': False})
    df = pd.read_csv(*args, **kwargs)
    if ((df.columns[0] in index_names or (df[df.columns[0]] == df.index).all()) or
        (df[df.columns[0]] == np.arange(len(df))).all() or
        ((df.index == np.arange(len(df))).all() and str(df[df.columns[0]].dtype).startswith('int') and
         df[df.columns[0]].count() == len(df))):
        df = df.set_index(df.columns[0], drop=True)
        if df.index.name in ('Unnamed: 0', ''):
            df.index.name = None
    try:
        df.index = pd.to_datetime(df.index)
    except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
        pass
    return df


for filename in CSVS:
    locals()['df_' + filename.split('.')[0].replace('-', '_')] = read_csv(os.path.join(DATA_PATH, filename))


harry_docs = ["The faster Harry got to the store, the faster and faster Harry would get home.",
              "Harry is hairy and faster than Jill.",
              "Jill is not as hairy as Harry."]


def no_tqdm(it, total=1):
    return it


def dropbox_basename(url):
    filename = os.path.basename(url)
    match = re.findall(r'\?dl=[0-9]$', filename)
    if match:
        return filename[:-len(match[0])]
    return filename


def download(names=None, verbose=True):
    names = [names] if isinstance(names, (str, bytes)) else names
    names = names or BIG_URLS.keys()
    file_paths = {}
    for name in names:
        name = name.lower().strip()
        if name in BIG_URLS:
            file_paths[name] = download_file(BIG_URLS[name][0],
                                             data_path=BIGDATA_PATH,
                                             size=BIG_URLS[name][1],
                                             verbose=verbose)
            if file_paths[name].endswith('.tar.gz'):
                untar(file_paths[name])
            file_paths[name] = file_paths[name][:-7]  # FIXME: rename tar.gz file so that it mimics contents
    return file_paths


def download_file(url, data_path=BIGDATA_PATH, filename=None, size=None, chunk_size=4096, verbose=True):
    """Uses stream=True and a reasonable chunk size to be able to download large (GB) files over https"""
    if filename is None:
        filename = dropbox_basename(url)
    file_path = os.path.join(data_path, filename)
    if url.endswith('?dl=0'):
        url = url[:-1] + '1'  # noninteractive download
    if verbose:
        tqdm_prog = tqdm
        print('requesting URL: {}'.format(url))
    else:
        tqdm_prog = no_tqdm
    r = requests.get(url, stream=True, allow_redirects=True)
    size = r.headers.get('Content-Length', None) if size is None else size
    print('remote size: {}'.format(size))

    stat = path_status(file_path)
    print('local size: {}'.format(stat.get('size', None)))
    if stat['type'] == 'file' and stat['size'] == size:  # TODO: check md5 or get the right size of remote file
        r.close()
        return file_path

    print('Downloading to {}'.format(file_path))

    with open(file_path, 'wb') as f:
        for chunk in tqdm_prog(r.iter_content(chunk_size=chunk_size)):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)

    r.close()
    return file_path


def multifile_dataframe(paths=['urbanslang{}of4.csv'.format(i) for i in range(1, 5)], header=0, index_col=None):
    """Like pandas.read_csv, but loads and concatenates (df.append(df)s) DataFrames together"""
    df = pd.DataFrame()
    for p in paths:
        df = df.append(read_csv(p, header=header, index_col=index_col), ignore_index=True if not index_col else False)
    if index_col and df.index.name == index_col:
        del df[index_col]
    return df


def read_json(file_path):
    return json.load(open(file_path, 'rt'))


def get_data(name='sms-spam'):
    try:
        return read_csv(os.path.join(DATA_PATH, name + '.csv.gz'))
    except IOError:
        pass
    try:
        return read_csv(os.path.join(DATA_PATH, name + '.csv'))
    except IOError:
        pass
    try:
        return read_json(os.path.join(DATA_PATH, name + '.json'))
    except IOError:
        pass
    msg = 'Unable to find dataset named {} in DATA_PATH with file extension .csv.gz, .csv, or .json'.format(name)
    logger.error(msg)
    raise IOError(msg)
