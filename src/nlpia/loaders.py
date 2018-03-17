""" Loaders and downloaders for data files and models required for the examples in NLP in Action

>>> get_data('cities_us').iloc[:3,:2]
        geonameid                           city
131484    4295856  Indian Hills Cherokee Section
137549    5322551                         Agoura
134468    4641562 
"""
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
from pugnlp.futil import path_status, find_files


import pandas as pd
import tarfile

from gensim.models import KeyedVectors

INT_MAX = INT64_MAX = 2 ** 63 - 1
INT_MIN = INT64_MIN = - 2 ** 63

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
        3112841563,
    ),
    'lsa_tweets_pickle': (
        'https://www.dropbox.com/s/7k0nvl2dx3hsbqp/lsa_tweets_5589798_2003588x200.pkl.projection.u.npy?dl=1',
        2990000000,
    ),
    'ubuntu_dialog': (
        'https://www.dropbox.com/s/krvi79fbsryytc2/ubuntu_dialog.csv.gz?dl=1',
        296098788,
    ),
    'imdb': (
        'https://www.dropbox.com/s/yviic64qv84x73j/aclImdb_v1.tar.gz?dl=1',
        3112841563,  # 3112841312,
    ),
}
BIG_URLS['word2vec'] = BIG_URLS['w2v']
DATA_NAMES = {
    'pointcloud': os.path.join(DATA_PATH, 'pointcloud.csv.gz'),
    'hutto_tweets0': os.path.join(DATA_PATH, 'hutto_ICWSM_2014/tweets_GroundTruth.csv.gz'),
    'hutto_tweets': os.path.join(DATA_PATH, 'hutto_ICWSM_2014/tweets_GroundTruth.csv'),
    'hutto_nyt': os.path.join(DATA_PATH, 'hutto_ICWSM_2014/nytEditorialSnippets_GroundTruth.csv.gz'),
    'hutto_movies': os.path.join(DATA_PATH, 'hutto_ICWSM_2014/movieReviewSnippets_GroundTruth.csv.gz'),
    'hutto_products': os.path.join(DATA_PATH, 'hutto_ICWSM_2014/amazonReviewSnippets_GroundTruth.csv.gz'),
}

DDL_DS_QUESTIONS_URL = 'http://minimum-entropy.districtdatalabs.com/api/questions/?format=json'
DDL_DS_ANSWERSS_URL = 'http://minimum-entropy.districtdatalabs.com/api/answers/?format=json'


W2V_PATH = os.path.join(BIGDATA_PATH, W2V_FILE)
TEXTS = ['kite_text.txt', 'kite_history.txt']
CSVS = ['mavis-batey-greetings.csv', 'sms-spam.csv']

DATA_INFO = pd.read_csv(os.path.join(DATA_PATH, 'data_info.csv'))


def untar(fname):
    if fname.endswith("tar.gz"):
        with tarfile.open(fname) as tf:
            tf.extractall()
    else:
        print("Not a tar.gz file: {}".format(fname))


for filename in TEXTS:
    with open(os.path.join(DATA_PATH, filename)) as fin:
        locals()[filename.split('.')[0]] = fin.read()
del fin


def read_csv(*args, **kwargs):
    """Like pandas.read_csv, only little smarter (checks first column to see if it should be the data frame index)

    >>> read_csv('mavis-batey-greetings.csv').head()
    """
    index_names = ('Unnamed: 0', 'pk', 'index', '')
    kwargs.update({'low_memory': False})
    if isinstance(args[0], pd.DataFrame):
        df = args[0]
    else:
        logger.info('Reading CSV with `read_csv(*{}, **{})`...'.format(args, kwargs))
        df = pd.read_csv(*args, **kwargs)
    if ((df.columns[0] in index_names or (df[df.columns[0]] == df.index).all()) or
        (df[df.columns[0]] == np.arange(len(df))).all() or
        ((df.index == np.arange(len(df))).all() and str(df[df.columns[0]].dtype).startswith('int') and
         df[df.columns[0]].count() == len(df))):
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


def read_txt(fin, nrows=None, verbose=True):
    lines = []
    if isinstance(fin, str):
        fin = open(fin)
    if verbose:
        tqdm_prog = tqdm
    else:
        tqdm_prog = no_tqdm
    with fin:
        for line in tqdm_prog(fin):
            lines += [line.rstrip('\n').rstrip('\r')]
            if nrows is not None and len(lines) >= nrows:
                break
        if all('\t' in line for line in lines):
            num_tabs = [sum([1 for c in line if c == '\t']) for line in lines]
            if all(i == num_tabs[0] for i in num_tabs):
                fin.seek(0)
                return read_csv(fin, sep='\t', header=None)
    return lines


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
    """ Download files or HTML tables listed in `names` and save them to DATA_PATH/`names`.csv

    Uses table in data_info.csv to determin URL or file path from name.

    TODO: if name is a valid URL then download it and create a name
          and store the name: url in data_info.csv before downloading
    """
    names = [names] if isinstance(names, (str, bytes)) else names
    # names = names or list(BIG_URLS.keys())  # download them all, if none specified!
    file_paths = {}
    for name in names:
        name = name.lower().strip()
        if name in BIG_URLS:
            file_paths[name] = download_file(BIG_URLS[name][0],
                                             data_path=BIGDATA_PATH,
                                             size=BIG_URLS[name][1],
                                             verbose=verbose)
            if file_paths[name].endswith('.tar.gz'):
                print('Extracting {}'.format(file_paths[name]))
                untar(file_paths[name])
                file_paths[name] = file_paths[name][:-7]  # FIXME: rename tar.gz file so that it mimics contents
        else:
            df = pd.read_html(DATA_INFO['url'][name], **DATA_INFO['downloader_kwargs'][name])[-1]
            df.columns = clean_columns(df.columns)
            file_paths[name] = os.path.join(DATA_PATH, name + '.csv')
            df.to_csv(file_paths[name])
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
    if stat['type'] == 'file' and stat['size'] >= size:  # TODO: check md5 or get the right size of remote file
        r.close()
        return file_path

    print('Downloading to {}'.format(file_path))

    with open(file_path, 'wb') as f:
        for chunk in tqdm_prog(r.iter_content(chunk_size=chunk_size)):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)

    r.close()
    return file_path


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

    # BIGDATA files are usually not loadable into dataframes
    try:
        return read_txt(os.path.join(BIGDATA_PATH, name + '.txt'), verbose=verbose)
    except IOError:
        pass
    try:
        return KeyedVectors.load_word2vec_format(os.path.join(BIGDATA_PATH, name + '.bin.gz'), binary=True)
    except IOError:
        pass
    except ValueError:
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
    >>> words[:8]
    ['A', "A's", "AA's", "AB's", "ABM's", "AC's", "ACTH's", "AI's"]
    """
    if name in BIG_URLS:
        print('Downloading {}'.format(name))
        filepaths = download(name)
        return filepaths[name]
    elif name in DATASET_NAME2FILENAME:
        return read_named_csv(name, data_path=DATA_PATH, nrows=nrows)
    elif name in DATA_NAMES:
        return read_named_csv(DATA_NAMES[name], nrows=nrows)
    elif os.path.isfile(name):
        return read_named_csv(name, nrows=nrows)
    elif os.path.isfile(os.path.join(DATA_PATH, name)):
        return read_named_csv(os.path.join(DATA_PATH, name), nrows=nrows)

    msg = 'Unable to find dataset named {} in DATA_PATH with file extension .csv.gz, .csv, .json, or .txt\n'.format(name)
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


def read_json(file_path):
    return json.load(open(file_path, 'rt'))


def get_wikidata_qnum(wikiarticle, wikisite):
    """Retrieve the Query number for a wikidata database of metadata about a particular article

    >>> print(get_wikidata_qnum(wikiarticle="Andromeda Galaxy", wikisite="enwiki"))
    Q2469
    """
    resp = requests.get('https://www.wikidata.org/w/api.php', {
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


def load_geonames(path='http://download.geonames.org/export/dump/cities1000.zip'):
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
    10 admin1 code       : fipscode (subject to change to iso code), see exceptions below, see file admin1Codes.txt
                           for display names of this code; varchar(20)
    11 admin2 code       : code for the second administrative division, a county in the US, see file admin2Codes.txt; varchar(80)
    12 admin3 code       : code for third level administrative division, varchar(20)
    13 admin4 code       : code for fourth level administrative division, varchar(20)
    14 population        : bigint (8 byte int)
    15 elevation         : in meters, integer
    16 dem               : digital elevation model, srtm3 or gtopo30, average elevation of 3''x3'' (ca 90mx90m) or 30''x30''
                           (ca 900mx900m) area in meters, integer. srtm processed by cgiar/ciat.
    17 timezone          : the iana timezone id (see file timeZone.txt) varchar(40)
    18 modification date : date of last modification in yyyy-MM-dd format
    """
    columns = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 'feature class', 'feature code', 'country code']
    columns += ['cc2', 'admin1_code', 'admin2_code', 'admin3_code', 'admin4_code', 'population', 'elevation', 'dem', 'timezone', 'modification date']
    columns = normalize_column_names(columns)
    df = pd.read_csv(path, sep='\t', index_col=None, low_memory=False, header=None)
    df.columns = columns
    return df


def normalize_column_names(df):
    columns = df.columns if hasattr(df, 'columns') else df
    columns = [c.lower().replace(' ', '_') for c in columns]
    return columns


def load_geo_adwords(filename='AdWords API Location Criteria 2017-06-26.csv.gz'):
    """There are still quite a few errors in this table, even after all this cleaning, so it's not a good source of city names"""
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
        logger.warn(('Loaded {} rows from tsv. Original file, "{}", contained {} seemingly valid lines.' +
                     'Index column: {}').format(len(df), original_len, filepath, index_col))
    return df
