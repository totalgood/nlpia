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
import regex
import os
import copy

from pugnlp.regexes import *  # noqa


logger = logging.getLogger(__name__)

RE_ACRONYM2 = r'((\w)[\w0-9]{2,16}[ ](\w)[\w0-9]{2,16})[ ]\((\2\3)\)'
RE_ACRONYM3 = r'((\w)[\w0-9]{2,16}[ ](\w)[\w0-9]{2,16}[ ](\w)[\w0-9]{2,16})[ ]\((\6\7\8)\)'
CRE_ACRONYM = re.compile(RE_ACRONYM2 + '|' + RE_ACRONYM3, re.IGNORECASE)

RE_URL_SIMPLE = r'(?P<url>(?P<scheme>(?P<scheme_type>http|ftp|https)://)?([^/:(\["\'`)\]\s]+' \
    r'[.])(com|org|edu|gov|net|mil|uk|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|io|me)([^"\'`)\]\s]*))'
CRE_URL_SIMPLE = re.compile(RE_URL_SIMPLE)
RE_URL_WITH_SCHEME = RE_URL_SIMPLE.replace('://)', '://)?')  # require scheme
CRE_URL_WITH_SCHEME = re.compile(RE_URL_WITH_SCHEME)

RE_HYPERLINK = RE_URL_WITH_SCHEME + r'\[(?P<name>[^\]]+)\]'
CRE_HYPERLINK = regex.compile(RE_HYPERLINK)
"""
>>> CRE_SLUG_DELIMITTER.sub('-', 'thisSlug-should|beHypenatedInLots_OfPlaces')
'this-Slug-should-be-Hypenated-In-Lots-Of-Places'
"""
CRE_SLUG_DELIMITTER = re.compile(r'[^a-zA-Z]+|(?<=[a-z])(?=[A-Z])')
"""
>>> CRE_FILENAME_EXT.search('~/.bashrc.asciidoc.ext.ps4.42').group()
'.asciidoc.ext.ps4.42'
>>> CRE_FILENAME_EXT.sub('', 'this/path/has/a/file.html')
'this/path/has/a/file'
>>> CRE_FILENAME_EXT.search('.bashrc..asciidoc.ext.ps4.123').group()
'.asciidoc.ext.ps4.123'
>>> CRE_FILENAME_EXT.search('.bashrc..asciidoc..ext.ps4.123').group()
'.ext.ps4.123'
"""
CRE_FILENAME_EXT = re.compile(r'(?<=[.a-zA-Z0-9_])([.][a-zA-Z0-9]{1,8}){1,5}$')


def splitext(filepath):
    """ Like os.path.splitext except splits compound extensions as one long one

    >>> splitext('~/.bashrc.asciidoc.ext.ps4.42')
    ('~/.bashrc', '.asciidoc.ext.ps4.42')
    >>> splitext('~/.bash_profile')
    ('~/.bash_profile', '')
    """
    exts = getattr(CRE_FILENAME_EXT.search(filepath), 'group', str)()
    return (filepath[:(-len(exts) or None)], exts)


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


class Pattern:
    """ Container for _regex.Pattern object augmented with Irregular matching rules 

    >>> pattern = Pattern('Aaron[ ]Swartz')
    >>> pattern.match('Aaron Swartz')
    <regex.Match object; span=(0, 12), match='Aaron Swartz'>
    >>> pattern.fullmatch('Aaron Swartz!!')
    >>> pattern.match('Aaron Swartz!!')
    <regex.Match object; span=(0, 12), match='Aaron Swartz'>
    """

    def __init__(self, pattern):
        pattern = getattr(pattern, 'pattern', pattern)
        self._compiled_pattern = pattern if hasattr(pattern, 'pattern') else regex.compile(pattern)
        for name in dir(self._compiled_pattern):
            if name in ('__class__', '__init__'):
                continue
            attr = getattr(self._compiled_pattern, name)
            try:
                setattr(self, name, attr)
                logger.debug('{}.{}.Pattern successfully "inherited" `_regex.Pattern.{}{}`'.format(
                    __package__, __name__, name, '()' if callable(attr) else ''))
            except:
                logger.warning('Unable to "inherit" `_regex.Pattern.{}{}`'.format(
                    name, '()' if callable(attr) else ''))


class REPattern:
    """ Container for re.SRE_Pattern object augmented with Irregular matching rules

    >>> pattern = REPattern('Aaron[ ]Swartz')
    >>> pattern.match('Aaron Swartz')
    <_sre.SRE_Match object; span=(0, 12), match='Aaron Swartz'>
    >>> pattern.fullmatch('Aaron Swartz!!')
    >>> pattern.fullmatch('Aaron Swartz')
    <regex.Match object; span=(0, 12), match='Aaron Swartz'>
    >>> pattern.match('Aaron Swartz!!')
    <_sre.SRE_Match object; span=(0, 12), match='Aaron Swartz'>
    """

    def __init__(self, pattern):
        self._compiled_pattern = re.compile(pattern)
        for name in dir(self._compiled_pattern):
            if name in ('__class__', '__init__', 'fullmatch') and getattr(self, name, None):
                continue
            attr = getattr(self._compiled_pattern, name)
            try:
                setattr(self, name, attr)
                logger.debug('{}.{}.{} successfully "inherited" `_regex.Pattern.{}{}`'.format(
                    __package__, __name__, self.__class__, name, '()' if callable(attr) else ''))
            except:
                logger.warning('Unable to "inherit" `_regex.Pattern.{}{}`'.format(
                    name, '()' if callable(attr) else ''))

    def fullmatch(self, *args, **kwargs):
        return regex.fullmatch(self._compiled_pattern.pattern, *args, **kwargs)
