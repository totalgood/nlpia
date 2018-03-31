#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

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
