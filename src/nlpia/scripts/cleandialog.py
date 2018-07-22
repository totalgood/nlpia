#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import (bytes, dict, int, list, object, range, str,  # noqa
    ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict,

import os
import argparse

from nlpia.data_utils import clean_csvs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess and line dialog CSV files in place.')
    parser.add_argument('-p', '--path', '--dialogpath', type=str, default='.', dest='dialogpath',
                        help='Path to directory of dialog CSV files or an individual dialog CSV file.')
    args = parser.parse_args()
    return args


def main(dialogpath=None):
    """ Parse the state transition graph for a set of dialog-definition tables to find an fix deadends """
    if dialogpath is None:
        args = parse_args()
        dialogpath = os.path.abspath(os.path.expanduser(args.dialogpath))
    else:
        dialogpath = os.path.abspath(os.path.expanduser(args.dialogpath))
    return clean_csvs(dialogpath=dialogpath)


if __name__ == '__main__':
    print(main())
