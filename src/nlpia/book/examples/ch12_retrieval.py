#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Natural Language Processing in Action -- Chapter 12 Getting Chatty -- 3. Retrieval (Search) section
"""
import os
import re

import tqdm
import aiml_bot

from nlpia.constants import DATA_PATH
from nlpia.data.loaders import get_data


def split_turns(s, splitter=re.compile('__eot__')):
    """ Split a string on __eot__ markders (turns) """
    for utterance in splitter.split(s):
        utterance = utterance.replace('__eou__', '\n')
        utterance = utterance.replace('__eot__', '')
        if len(utterance.strip()):
            yield utterance


def preprocess_ubuntu_corpus(df):
    """Split all strings in df.Context and df.Utterance on __eot__ (turn) markers """
    statements = []
    replies = []
    for i, record in tqdm(df.iterrows()):
        turns = list(split_turns(record.Context))
        statement = turns[-1] if len(turns) else '\n'  # <1>
        statements.append(statement)
        turns = list(split_turns(record.Utterance))
        reply = turns[-1] if len(turns) else '\n'
        replies.append(reply)
    df['statement'] = statements
    df['reply'] = replies
    return df


def format_ubuntu_dialog(df):
    """ Print statements paired with replies, formatted for easy review """
    s = ''
    for i, record in df.iterrows():
        statement = list(split_turns(record.Context))[-1]  # <1>
        reply = list(split_turns(record.Utterance))[-1]  # <2>
        s += 'Statement: {}\n'.format(statement)
        s += 'Reply: {}\n\n'.format(reply)
    return s
    # <1> We need to use `list` to force iteration through the generator
    # <2> The `[-1]` index retrievs the last "turn" in the sequence, discarding everything else


if __name__ == '__main__':
    df = get_data('ubuntu_dialog')
    df = preprocess_ubuntu_corpus(df)
    print(format_ubuntu_dialog(df.head(4)))
