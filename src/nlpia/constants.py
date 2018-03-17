from __future__ import print_function, unicode_literals, division, absolute_import
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict,
from builtins import *  # noqa

import logging
import logging.config
import os

import configparser

from pugnlp.util import dict2obj
import platform

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

system_name = platform.system()
if system_name == 'Darwin':
    SYSLOG_PATH = os.path.join(os.path.sep, 'var', 'run', 'syslog')
elif system_name == 'Linux':
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
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'basic',
            'stream': 'ext://sys.stdout',
        },
    },

    'loggers': {
        'loggly': {
            'handlers': ['console'],
            'propagate': True,
            'format': 'django: %(message)s',
            'level': 'DEBUG',
        },
    },
}


# Set up syslogger for loggly service if the /dev socket exists or use NTEventLogHandler on Windows (no syslog /dev).
if system_name == 'Windows':
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


logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

USER_HOME = os.path.expanduser("~")
PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
BIGDATA_PATH = os.path.join(os.path.dirname(__file__), 'bigdata')

# rename secrets.cfg.EXAMPLE_TEMPLATE -> secrets.cfg then edit secrets.cfg to include your actual credentials
secrets = configparser.RawConfigParser()
try:
    secrets.read(os.path.join(PROJECT_PATH, 'secrets.cfg'))
    secrets = secrets._sections
except IOError:
    logger.error('Unable to load/parse secrets.cfg file at "{}". Does it exist?'.format(os.path.join(PROJECT_PATH, 'secrets.cfg')))
    secrets = {}

secrets = dict2obj(secrets)
