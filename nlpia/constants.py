import logging
import logging.config
import os

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'django': {
            'format': 'django: %(message)s',
        },
        u'basic': {
            u'format': u'%(asctime)s | %(name)15s:%(lineno)3s:%(funcName)15s | %(levelname)7s | %(message)s',
        }
    },

    'handlers': {
        'logging.handlers.SysLogHandler': {
            'level': 'DEBUG',
            'class': 'logging.handlers.SysLogHandler',
            'facility': 'local7',
            'formatter': 'django',
            'address': '/dev/log',
        },
        u'console': {
            u'class': u'logging.StreamHandler',
            u'level': u'DEBUG',
            u'formatter': u'basic',
            u'stream': u'ext://sys.stdout',
        },
    },

    'loggers': {
        'loggly': {
            'handlers': [u'console', 'logging.handlers.SysLogHandler'],
            'propagate': True,
            'format': 'django: %(message)s',
            'level': 'DEBUG',
        },
    },
}

logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)


USER_HOME = os.path.expanduser("~")
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
DATA_URL = 'http://totalgood.org/static/data'
W2V_FILE = 'GoogleNews-vectors-negative300.bin.gz'
W2V_URL = 'https://www.dropbox.com/s/4bcegydk3pn9067/GoogleNews-vectors-negative300.bin.gz?dl=0'
BIG_URLS = {'w2v': W2V_URL,
            'slang': 'https://www.dropbox.com/s/51mfq87zvct1t1y/slang.csv.gz?dl=0'}
W2V_PATH = os.path.join(DATA_PATH, W2V_FILE)
TEXTS = ['kite_text.txt', 'kite_history.txt']
CSVS = ['mavis-greeting-training-set.csv']
