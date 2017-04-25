from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import int, round, str
from future import standard_library
standard_library.install_aliases()
from builtins import object  # NOQA

from gensim.models import Word2Vec

class W2V(object):
    w2v = None

    def __init__(self, path='/data/Googl'):

