#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load sentiment analysis word value labels from academic studies

senti # large dataframe for sentiment and stats on about 14k word lemmas
senti_classic  # original sentiment data used for VADER and other "classic" algorithms

>>> wisconsin.head().iloc[:,:5]
        word  sound_mean  sound_n  color_mean  color_n
2   accident        4.00       32        0.94       32
3     accord        0.67       30        0.19       31
4  accordion        4.66       32        2.44       32
5    acrobat        1.31       32        2.19       32
6  actuality        0.52       31        0.00       32

>>> warriner.head().iloc[:,:5]
          Word  V_Mean_Sum  V_SD_Sum  V_Rat_Sum  A_Mean_Sum
0     aardvark        6.26      2.21         19        2.41
1      abalone        5.30      1.59         20        2.65
2      abandon        2.84      1.54         19        3.73
3  abandonment        2.63      1.74         19        4.95
4        abbey        5.85      1.69         20        2.20
"""
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import (bytes, dict, int, list, object, range, str,  # noqa
    ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict,

import os
from itertools import product

import pandas as pd

from nlpia.constants import DATA_PATH
from nlpia.loaders import get_data


wisconsin = get_data('sentiment-word-ratings-sensori-motor-wisconsin.csv.gz')
warriner = get_data('sentiment-word-ratings-warriner.csv.gz')


def get_sentiment_sensori_motor():
    df = pd.read_html('http://www.neuro.mcw.edu/ratings/')[0]
    df.columns = ['word'] + [i.lower() + "_" + j.lower() for i, j in product(df.iloc[0][1:6], df.iloc[1][1:3])]
    df = df.iloc[2:]
    df.to_csv(os.path.join(DATA_PATH, 'sentiment-word-ratings-sensori-motor-wisconsin.csv.gz'), compression='gzip')
    return df


def get_sentiment_warriner():
    senti = pd.read_csv('http://crr.ugent.be/papers/Ratings_Warriner_et_al.csv', index_col='Word', header=0)
    senti.columns = [c.replace('.', '_') for c in senti.columns]
    del senti['Unnamed: 0']
    senti.to_csv(os.path.join(DATA_PATH, 'sentiment-word-ratings-warriner.csv.gz'), compression='gzip')
    return senti
