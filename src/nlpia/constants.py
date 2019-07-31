# -*- coding: utf-8 -*-
""" Constants and global configuration options, like `logging.getLogger` and loading secrets.cfg """
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import (bytes, dict, int, list, object, range, str, ascii, chr,  # noqa
                      hex, input, next, oct, open, pow, round, super, filter, map, zip)
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict,

import configparser
import logging
import logging.config
import os
import errno
from collections import Mapping

from pandas import read_csv
from tqdm import tqdm  # noqa

from pugnlp.futil import touch_p
import platform

REQUESTS_HEADER = (
    ('User-Agent', 'Mozilla Firefox'),
    ('From', 'nlpia+github@totalgood.com'),
    ('Referer', 'http://github.com/totalgood/nlpia'),
    )

LOG_LEVEL = 'WARN' if not os.environ.get('DEBUG') else 'DEBUG'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM_NAME = platform.system()
if SYSTEM_NAME == 'Darwin':
    SYSLOG_PATH = os.path.join(os.path.sep, 'var', 'run', 'syslog')
elif SYSTEM_NAME == 'Linux':
    SYSLOG_PATH = os.path.join('dev', 'log')
else:
    SYSLOG_PATH = None
if SYSLOG_PATH and not os.path.exists(SYSLOG_PATH):
    SYSLOG_PATH = None


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'django': {
            'format': 'django: %(message)s',
        },
        'basic': {
            'format': '%(asctime)s %(levelname)7s:%(name)15s:%(lineno)3s:%(funcName)20s %(message)s',
        },
        'short': {
            'format': '%(asctime)s %(levelname)s:%(name)s:%(message)s'
        },
    },
    'handlers': {
        'default': {
            'class': 'logging.StreamHandler',
            'level': LOG_LEVEL,
            'formatter': 'basic',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': LOG_LEVEL,
            'propagate': True,
        },
        'loggly': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}


# Set up syslogger for loggly service if the /dev socket exists or use NTEventLogHandler on Windows (no syslog /dev).
if SYSTEM_NAME == 'Windows':
    LOGGING_CONFIG['loggers']['loggly']['handlers'] += ['logging.handlers.NTEventLogHandler']
    LOGGING_CONFIG['handlers']['logging.handlers.NTEventLogHandler'] = {
        'level': 'DEBUG',
        'class': 'logging.handlers.NTEventLogHandler',
        'formatter': 'django'
    }
elif SYSLOG_PATH:
    LOGGING_CONFIG['loggers']['loggly']['handlers'] += ['logging.handlers.SysLogHandler']
    LOGGING_CONFIG['handlers']['logging.handlers.SysLogHandler'] = {
        'level': 'DEBUG',
        'class': 'logging.handlers.SysLogHandler',
        'facility': 'local7',
        'formatter': 'django',
        'address': SYSLOG_PATH,
    }


try:
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    raise NotImplementedError("Force logger to fall back to failsafe file logging.")
except:  # noqa
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
logger.warning('Starting logger in nlpia.constants...')

USER_HOME = os.path.expanduser("~")
PROJECT_PATH = PRJECT_DIR = BASE_DIR

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
BOOK_PATH = os.path.join(DATA_PATH, 'book')
DATA_INFO_FILE = os.path.join(DATA_PATH, 'data_info.csv')

BIGDATA_PATH = os.path.join(os.path.dirname(__file__), 'bigdata')
BIGDATA_INFO_FILE = os.path.join(DATA_PATH, 'bigdata_info.csv')
BIGDATA_INFO_LATEST = BIGDATA_INFO_FILE[:-4] + '.latest.csv'
touch_p(BIGDATA_INFO_FILE, times=False)
touch_p(BIGDATA_INFO_LATEST, times=False)
CHECKPOINT_PATH = os.path.join(BIGDATA_PATH, 'checkpoints')

UTF8_TABLE = read_csv(os.path.join(DATA_PATH, 'utf8.csv'))
UTF8_TO_MULTIASCII = dict(zip(UTF8_TABLE.char, UTF8_TABLE.multiascii))
UTF8_TO_ASCII = dict(zip(UTF8_TABLE.char, UTF8_TABLE.ascii))

INT_MAX = INT64_MAX = 2 ** 63 - 1
INT_MIN = INT64_MIN = - 2 ** 63
INT_NAN = INT64_NAN = INT64_MIN
INT_MIN = INT64_MIN = INT64_MIN + 1

MIN_DATA_FILE_SIZE = 100  # loaders.get_data() will fail on files < 100 bytes
MAX_LEN_FILEPATH = 1023  # on OSX `open(fn)` raises OSError('Filename too long') if len(fn)>=1024

HTML_TAGS = '<HTML', '<A HREF=', '<P>', '<BOLD>', '<SCRIPT', '<DIV', '<TITLE', '<BODY', '<HEADER'
EOL = os.linesep


def mkdir_p(path, exist_ok=True):
    """ mkdir -p functionality (make intervening directories and ignore existing directories)

    SEE: https://stackoverflow.com/a/600612/623735

    >>> deeper_path = os.path.join(BIGDATA_PATH, 'doctest_nlpia', 'constants', 'mkdir_p')
    >>> mkdir_p(deeper_path, exist_ok=False)
    >>> os.path.isdir(deeper_path)
    True
    >>> mkdir_p(deeper_path, exist_ok=True)
    >>> os.path.isdir(deeper_path)
    True
    """
    path = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path) and exist_ok:
            pass
        else:
            raise


class Object(object):
    """If your dict is "flat", this is a simple way to create an object from a dict

    >>> obj = Object()
    >>> obj.__dict__ = {'a': 1, 'b': 2}
    >>> obj.a, obj.b
    (1, 2)
    """
    pass


# For a nested dict, you need to recursively update __dict__
def dict2obj(d):
    """Convert a dict to an object or namespace


    >>> d = {'a': 1, 'b': {'c': 2}, 'd': ["hi", {'foo': "bar"}]}
    >>> obj = dict2obj(d)
    >>> obj.b.c
    2
    >>> obj.d
    ['hi', {'foo': 'bar'}]
    >>> d = {'a': 1, 'b': {'c': 2}, 'd': [("hi", {'foo': "bar"})]}
    >>> obj = dict2obj(d)
    >>> obj.d.hi.foo
    'bar'
    """
    if isinstance(d, (Mapping, list, tuple)):
        try:
            d = dict(d)
        except (ValueError, TypeError):
            return d
    else:
        return d
    obj = Object()
    for k, v in d.items():
        obj.__dict__[k] = dict2obj(v)
    return obj


def no_tqdm(it, total=1, **kwargs):
    """ Do-nothing iterable wrapper to subsitute for tqdm when verbose==False """
    return it


if not os.path.isdir(BIGDATA_PATH):
    mkdir_p(BIGDATA_PATH, exist_ok=True)
if not os.path.isdir(CHECKPOINT_PATH):  # Thank you Matt on livebook for catching this
    mkdir_p(CHECKPOINT_PATH, exist_ok=True)


# rename secrets.cfg.EXAMPLE_TEMPLATE -> secrets.cfg then edit secrets.cfg to include your actual credentials
secrets = configparser.RawConfigParser()
try:
    secrets.read(os.path.join(PROJECT_PATH, 'secrets.cfg'))
    secrets = secrets._sections
except IOError:
    logger.error('Unable to load/parse secrets.cfg file at "%s". Does it exist?',
                 os.path.join(PROJECT_PATH, 'secrets.cfg'))
    secrets = {}

secrets = dict2obj(secrets)
