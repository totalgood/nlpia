from __future__ import print_function, unicode_literals, division, absolute_import
from future import standard_library
standard_library.install_aliases() # noqa
from builtins import object  # noqa

# from gensim.models import Word2Vec
from gensim import corpora
from gensim import utils

from nlpia.constants import logging
logger = logging.getLogger(__name__)


def tokens2ngrams(tokens, n=2):
    tokens = list(tokens)
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(' '.join(tokens[i:i + n]))
    return ngrams


def passthrough(*args, **kwargs):
    return args[0] if len(args) else list(kwargs.values())[0]


def return_false(*args, **kwargs):
    return False


def return_true(*args, **kwargs):
    return True


def noop(*args, **kwargs):
    pass


def return_none(*args, **kwargs):
    pass


def to_unicode(s_or_b):
    if isinstance(s_or_b, bytes):
        s_or_b = s_or_b.decode('utf-8')
    return utils.to_unicode(s_or_b)


class TweetCorpus(corpora.TextCorpus):
    ignore_matcher = return_none   # compiled regular expression for token matches to skip/ignore
    num_grams = 2
    case_normalizer = str
    tokenizer = None
    mask = None

    def get_texts(self):
        """ Parse documents from a .txt file assuming 1 document per line, yielding lists of filtered tokens """
        with self.getstream() as text_stream:
            for i, line in enumerate(text_stream):
                line = to_unicode(line)
                print(line)
                # line = self.case_normalizer(line)
                if self.mask is not None and not self.mask[i]:
                    continue
                ngrams = []
                for ng in tokens2ngrams((TweetCorpus.tokenizer or str.split)(line), n=self.num_grams):
                    if self.ignore_matcher(ng):
                        continue
                    ngrams += [ng]
                yield ngrams

    def __len__(self):
        """ Enables `len(corpus)` """
        if 'length' not in self.__dict__:
            logger.info("Computing the number of lines in the corpus size (calculating number of documents)")
            self.length = sum(1 for doc in self.get_stream())
        return self.length


class SMSCorpus(corpora.TextCorpus):
    ignore_matcher = return_none   # compiled regular expression for token matches to skip/ignore
    num_grams = 2
    case_normalizer = utils.to_unicode
    tokenizer = str.split
    mask = None

    def get_texts(self):
        """ Parse documents from a .txt file assuming 1 document per line, yielding lists of filtered tokens """
        with self.getstream() as text_stream:
            for i, line in enumerate(text_stream):
                line = self.case_normalizer(line)
                print(i, line)
                if self.mask is not None and not self.mask[i]:
                    continue
                ngrams = []
                for ng in tokens2ngrams(self.tokenizer(line)):
                    if self.ignore_matcher(ng):
                        continue
                    ngrams += [ng]
                yield ngrams

    def __len__(self):
        """ Enables `len(corpus)` """
        if 'length' not in self.__dict__:
            logger.info("Computing the number of lines in the corpus size (calculating number of documents)")
            self.length = sum(1 for doc in self.getstream())
        return self.length
