# -*- coding: utf-8 -*-
""" Compiled regular expressions for extracting dates, times, acronyms, etc

>>> CRE_ACRONYM.findall('National science Foundation (NSF)')
[('', '', '', '', 'National science Foundation', 'N', 's', 'F', 'NSF')]
>>> re.findall(RE_URL_SIMPLE, '* Sublime Text 3 (https://www.sublimetext.com/3) is great!')[0][0]
'https://www.sublimetext.com/3'
>>> re.findall(RE_URL_SIMPLE, 'Google github totalgood [github.com/totalgood]!')[0][0]
'github.com/totalgood'
"""
from nlpia.constants import logging, DATA_PATH
import re
import os
import copy

from pugnlp.regexes import *  # noqa


logger = logging.getLogger(__name__)

RE_ACRONYM2 = r'((\w)[\w0-9]{2,16}[ ](\w)[\w0-9]{2,16})[ ]\((\2\3)\)'
RE_ACRONYM3 = r'((\w)[\w0-9]{2,16}[ ](\w)[\w0-9]{2,16}[ ](\w)[\w0-9]{2,16})[ ]\((\6\7\8)\)'
CRE_ACRONYM = re.compile(RE_ACRONYM2 + '|' + RE_ACRONYM3, re.IGNORECASE)

RE_URL_SIMPLE = r'(((http|ftp|https)://)?([^/:(\["\'`)\]\s]+' \
    r'[.])(com|org|edu|gov|net|mil|uk|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|io|me)([^"\'`)\]\s]*))'
CRE_URL_SIMPLE = re.compile(RE_URL_SIMPLE)

"""
>>> CRE_SLUG_DELIMITTER.sub('-', 'thisSlug-should|beHypenatedInLots_OfPlaces')
'this-Slug-should-be-Hypenated-In-Lots-Of-Places'
"""
CRE_SLUG_DELIMITTER = re.compile(r'[^a-zA-Z]+|(?<=[a-z])(?=[A-Z])')


CRE_FILENAME_EXT = re.compile(r'(?<=[.a-zA-Z0-9_])([.][a-zA-Z0-9]{1,8}){1,5}$')
"""
>>> CRE_FILENAME_EXT.search('.bashrc.asciidoc.ext.ps4.123').group()
''
>>> CRE_FILENAME_EXT.sub('', 'this/path/has/a/file.html')
'this/path/has/a/file'
>>> CRE_FILENAME_EXT.search('.bashrc..asciidoc.ext.ps4.123').group()
'.asciidoc.ext.ps4.123'
>>> CRE_FILENAME_EXT.search('.bashrc..asciidoc..ext.ps4.123').group()
'.ext.ps4.123'
"""

# ? \(\): ()
# \': '"'"' 
# \s: [:space:]
# RE_URL_BASH_ESCAPE = '((http|ftp|https)://)?[^/:\(\[\"'"'"'\`\)\] \t\n]+[.](com|org|edu|gov|net|mil|uk|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|io|me)[^\"'"'"'\`\)\] \t\n]*'  # noqa


def to_tsv():
    """ Save all regular expressions to a tsv file so they can be more easily copy/pasted in Sublime """
    with open(os.path.join(DATA_PATH, 'regexes.tsv'), mode='wt') as fout:
        vars = copy.copy(tuple(globals().items()))
        for k, v in vars:
            if k.lower().startswith('cre_'):
                fout.write(k[4:] + '\t' + v.pattern + '\n')
            elif k.lower().startswith('re_'):
                fout.write(k[3:] + '\t' + v.pattern + '\n')
