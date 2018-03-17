"""Load sentiment analysis word value labels from academic studies

senti # large dataframe for sentiment and stats on about 14k word lemmas
senti_classic  # original sentiment data used for VADER and other "classic" algorithms

>>> senti_sensori_motor.describe()
        word sound_mean sound_n color_mean color_n manipulation_mean  \
count   1402       1402    1402       1402    1402              1402
unique  1402        424      28        423      24               343
top      bye       1.00      11       2.00      12              2.00
freq       1         23     157         23     180                31
       manipulation_n motion_mean motion_n emotion_mean emotion_n
count            1402        1402     1402         1402      1402
unique             25         376       29          558        22
top                13        1.00       12         0.00        12
freq              229          27      161           17       265

>>> senti.describe()
         Unnamed: 0    V.Mean.Sum      V.SD.Sum     V.Rat.Sum    A.Mean.Sum  \
count  13915.000000  13915.000000  13915.000000  13915.000000  13915.000000
mean    6958.000000      5.063847      1.679261     21.812433      4.210582
std     4017.058833      1.274892      0.350281     23.441875      0.896413
min        1.000000      1.260000      0.310000     16.000000      1.600000
25%     3479.500000      4.250000      1.430000     19.000000      3.560000
50%     6958.000000      5.200000      1.670000     20.000000      4.110000
75%    10436.500000      5.950000      1.910000     21.000000      4.760000
max    13915.000000      8.530000      3.290000    872.000000      7.790000
           A.SD.Sum     A.Rat.Sum    D.Mean.Sum      D.SD.Sum     D.Rat.Sum  \
count  13915.000000  13915.000000  13915.000000  13915.000000  13915.000000
mean       2.300198     22.974057      5.184773      2.159786     24.315128
std        0.320252     24.726507      0.938284      0.328592     25.066800
min        0.880000     16.000000      1.680000      0.780000     14.000000
25%        2.080000     20.000000      4.580000      1.940000     20.000000
50%        2.300000     21.000000      5.260000      2.170000     22.000000
75%        2.520000     23.000000      5.840000      2.380000     25.000000
max        3.360000    919.000000      7.900000      3.370000    980.000000
           ...            A.Rat.L      A.Mean.H        A.SD.H       A.Rat.H  \
count      ...       13915.000000  13915.000000  13914.000000  13915.000000
mean       ...          12.733956      4.284328      2.171885     10.240101
std        ...          13.958102      1.093094      0.514230     11.379156
min        ...           5.000000      1.120000      0.000000      1.000000
25%        ...          10.000000      3.500000      1.850000      8.000000
50%        ...          11.000000      4.200000      2.190000     10.000000
75%        ...          14.000000      5.000000      2.510000     12.000000
max        ...         509.000000      8.670000      4.620000    411.000000
           D.Mean.L        D.SD.L       D.Rat.L      D.Mean.H        D.SD.H  \
count  13915.000000  13915.000000  13915.000000  13915.000000  13915.000000
mean       5.196908      2.218375     13.237657      5.170126      2.024414
std        1.051368      0.465165     13.913171      1.074364      0.504523
min        1.670000      0.520000      5.000000      1.330000      0.000000
25%        4.500000      1.900000      9.000000      4.500000      1.670000
50%        5.250000      2.220000     12.000000      5.250000      2.030000
75%        5.920000      2.540000     16.000000      5.920000      2.380000
max        8.570000      3.850000    534.000000      8.670000      4.620000
            D.Rat.H
count  13915.000000
mean      11.077470
std       11.675659
min        2.000000
25%        9.000000
50%       10.000000
75%       12.000000
max      447.000000
"""

from __future__ import print_function, unicode_literals, division, absolute_import
from future import standard_library
standard_library.install_aliases()  # noqa
from builtins import *  # noqa

import os
from itertools import product

import pandas as pd

from nlpia.constants import DATA_PATH


df = pd.read_html('http://www.neuro.mcw.edu/ratings/')[0]
df.columns = ['word'] + [i.lower() + "_" + j.lower() for i, j in product(df.iloc[0][1:6], df.iloc[1][1:3])]
df = df.iloc[2:]
df.to_csv(os.path.join(DATA_PATH, 'sentiment-word-ratings-sensori-motor-wisconsin.csv.gz'), compression='gzip')
senti_sensori_motor = df

senti = pd.read_csv('http://crr.ugent.be/papers/Ratings_Warriner_et_al.csv', index_col='Word', header=0)
senti.columns = [c.replace('.', '_') for c in senti.columns]
del senti['Unnamed: 0']
senti.to_csv(os.path.join(DATA_PATH, 'sentiment-word-ratings-warriner.csv.gz'), compression='gzip')
