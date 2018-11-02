""" Example python snippets and listing in Chapter 6 """
import pandas as pd
import numpy as np
from nlpia.loaders import get_data

wv = get_data('word2vec')

"""
>>> answer_vector = wv['woman'] + wv['Europe'] + wv[physics'] +\
...     wv['scientist']
"""

answer_vector = wv['woman'] + wv['Europe'] + wv['physics'] + wv['scientist']