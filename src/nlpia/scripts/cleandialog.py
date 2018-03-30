#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer
import os
from traceback import format_exc

import argparse

from nlpia.data_utils import clean_df


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess and line dialog CSV files in place.')
    parser.add_argument('-p', '--path', '--dialogpath', type=str, default='.', dest='dialogpath',
                        help='Path to directory of dialog CSV files or an individual dialog CSV file.')
    args = parser.parse_args()
    return args


def main(dialogpath=None):
    if dialogpath is None:
        args = parse_args()
        dialogpath = os.path.abspath(os.path.expanduser(args.dialogpath))
    else:
        dialogpath = os.path.abspath(os.path.expanduser(args.dialogpath))
    dialogdir = os.dirname(dialogpath) if os.path.isfile(dialogpath) else dialogpath
    filenames = [dialogpath.split(os.path.sep)[-1]] if os.path.isfile(dialogpath) else os.listdir(dialogpath)
    for filename in filenames:
        filepath = os.path.join(dialogdir, filename)
        df = clean_df(filepath)
        df.to_csv(filepath, header=0)
    return filenames


if __name__ == '__main__':
    print(main())
