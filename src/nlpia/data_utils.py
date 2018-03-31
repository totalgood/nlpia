from __future__ import absolute_import, division, print_function, unicode_literals
from future import standard_library
standard_library.install_aliases()  # noqa
from builtins import *  # noqa
import tempfile
import os

import pandas as pd
from nlpia.constants import logging

from annoy import AnnoyIndex

from nlpia.constants import UTF8_TO_ASCII, UTF8_TO_MULTIASCII
from nlpia.data.loaders import read_csv


np = pd.np
logger = logging.getLogger(__name__)


def clean_csvs(dialogpath=None):
    """ Translate non-ASCII characters to spaces or equivalent ASCII characters """
    dialogdir = os.dirname(dialogpath) if os.path.isfile(dialogpath) else dialogpath
    filenames = [dialogpath.split(os.path.sep)[-1]] if os.path.isfile(dialogpath) else os.listdir(dialogpath)
    for filename in filenames:
        filepath = os.path.join(dialogdir, filename)
        df = clean_df(filepath)
        df.to_csv(filepath, header=None)
    return filenames


def unicode2ascii(text, expand=True):
    r""" Translate UTF8 characters to ASCII

    >>> unicode2ascii("żółw")
    zozw

    utf8_letters =  'ą ę ć ź ż ó ł ń ś “ ” ’'.split()
    ascii_letters = 'a e c z z o l n s " " \''
    """
    translate = UTF8_TO_ASCII if not expand else UTF8_TO_MULTIASCII
    output = ''
    for c in text:
        if not c or ord(c) < 128:
            output += c
        else:
            output += translate[c] if c in translate else ' '
    return output.strip()


def clean_df(df, header=None, **read_csv_kwargs):
    """ Convert UTF8 characters in a CSV file or dataframe into ASCII

    Args:
      df (DataFrame or str): DataFrame or path or url to CSV
    """
    df = read_csv(df, header=header, **read_csv_kwargs)
    df = df.fillna(' ')
    for col in df.columns:
        df[col] = df[col].apply(unicode2ascii)
    return df


def representative_sample(X, num_samples, save=False):
    """Sample vectors in X, prefering edge cases and vectors farthest from other vectors in sample set


    """
    X = X.values if hasattr(X, 'values') else np.array(X)
    N, M = X.shape
    rownums = np.arange(N)
    np.random.shuffle(rownums)

    idx = AnnoyIndex(M)
    for i, row in enumerate(X):
        idx.add_item(i, row)
    idx.build(int(np.log2(N)) + 1)

    if save:
        if isinstance(save, (bytes, str)):
            idxfilename = save
        else:
            idxfile = tempfile.NamedTemporaryFile(delete=False)
            idxfile.close()
            idxfilename = idxfile.name
        idx.save(idxfilename)
        idx = AnnoyIndex(M)
        idx.load(idxfile.name)

    samples = -1 * np.ones(shape=(num_samples,), dtype=int)
    samples[0] = rownums[0]
    # FIXME: some integer determined by N and num_samples and distribution
    j, num_nns = 0, min(1000, int(num_samples / 2. + 1))
    for i in rownums:
        if i in samples:
            continue
        nns = idx.get_nns_by_item(i, num_nns)
        # FIXME: pick vector furthest from past K (K > 1) points or outside of a hypercube 
        #        (sized to uniformly fill the space) around the last sample
        try:
            samples[j + 1] = np.setdiff1d(nns, samples)[-1]
        except:
            samples[j + 1]
        if len(num_nns) < num_samples / 3.:
            num_nns = min(N, 1.3 * num_nns)
        j += 1
    return samples
