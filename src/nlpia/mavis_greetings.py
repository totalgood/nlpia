#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Constants and discovered values, like path to current installation of pug-nlp."""
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import (bytes, dict, int, list, object, range, str,  # noqa
    ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict,

import os
import pandas as pd
from pugnlp.constants import DATA_PATH


if __name__ == '__main__':
    df = pd.DataFrame()
    for is_greeting, filename in enumerate(['mavis-batey-sentences.txt', 'mavis-batey-greetings.txt']):
        with open(os.path.join(DATA_PATH, filename)) as f:
            df = pd.concat([df, pd.DataFrame([[sentence.strip(), is_greeting] for sentence in f],
                                             columns=['sentence', 'is_greeting'])],
                           ignore_index=True)

    df.to_csv(os.path.join(DATA_PATH, 'mavis-greeting-training-set.csv'))
    # df = pd.DataFrame.from_csv(
    # 'https://raw.githubusercontent.com/totalgood/pugnlp/master/pugnlp/data/mavis-greeting-training-set.csv',
    # header=0)
