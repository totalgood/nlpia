# -*- coding: utf-8 -*-
""" Compiled regular expressions for extracting dates, times, acronyms, etc

>>> m = CRE_ACRONYM.finditer('National science Foundation (NSF) arti-ficial intelligence (ai)')
>>> m = list(m)
>>> m[0]['a3']
'NSF'
>>> m[0]['a2']
>>> m[1]['a2']
'ai'
>>> CRE_ACRONYM.findall('National Science Foundation (NSF)')
[('', '', '', '', 'National Science Foundation', 'N', 'S', 'F', 'NSF' ...)]
>>> CRE_ACRONYM.findall('A Long iffy chatbot eponym (A.L.I.C.E.)')
[(...'A Long iffy chatbot eponym', ... 'A.L.I.C.E.' ...)]

>>> pat = r'^({italicstart}{w}{{5,10}}{italicend}[ ]){{2}}$'.format(**PATTERNS)
>>> pat = re.compile(pat)
>>> pat.match('_italicized_ _words_ ')
<_sre.SRE_Match object; span=(0, 21), match='_italicized_ _words_ '>
"""
from nlpia.constants import logging
import re


logger = logging.getLogger(__name__)

# kind of like stopwords, but just the words that are commonly lowercased in article titles
TITLE_LOWERWORDS = sorted('of a in the on as if and or but with'.split())
RE_ACRONYM_IGNORE = '(?:' + '|'.join(TITLE_LOWERWORDS) + ')'

RE_BREAK_CHARCLASS = r'\b[^-_a-zA-Z0-9]'  # like \W but doesn't allow "-" to break words
RE_STYLEMARK = r'[_*+^~]'  # italics, bold, math, superscript, subscript

RE_BOLD_START = r'(?:(?<![*])\*(?=[a-zA-Z0-9]))'  # start delimiter for bolded word
RE_BOLD_END = r'(?:\*(?![*]))'  # end delimiter for bolded word
RE_BOLD_CHAR_START = r'(?:(?<![*])\*\*(?=[a-zA-Z0-9]))'  # start delimiter for single character bolded
RE_BOLD_CHAR_END = r'(?:(?=[a-zA-Z0-9])\*\*(?![*]))'  # end delimiter for single character bolded

RE_ITALIC_START = r'(?:(?<!_)_(?=[a-zA-Z0-9]))'  # start delimiter for italicized word
RE_ITALIC_END = r'(?:_(?!_))'  # end delimiter for italicized word
RE_ITALIC_CHAR_START = r'(?:(?<!_)__(?=[a-zA-Z0-9]))'  # start delimiter for single character italicized
RE_ITALIC_CHAR_END = r'(?:(?=[a-zA-Z0-9])__(?!_))'  # end delimiter for single character italicized

RE_WORD_CHARCLASS = r'[-a-zA-Z0-9]'  # like \w but for English, not code, so "-" allowed but not "_"
RE_OPTIONAL_WORD = '(?:' + RE_WORD_CHARCLASS + '{0,16})'  # like \w but for English, not code, so "-" allowed but not "_"
RE_ENGLISH_WORD = '(?:' + RE_WORD_CHARCLASS + '{1,16})'

RE_STYLE_START = '(?:' + '|'.join(
    [RE_BOLD_START, RE_BOLD_CHAR_START, RE_ITALIC_START, RE_ITALIC_CHAR_START]
    ) + ')'
RE_STYLE_END = '(?:' + '|'.join(
    [RE_BOLD_END, RE_BOLD_CHAR_END, RE_ITALIC_END, RE_ITALIC_CHAR_END]
    ) + ')'

PATTERNS = {
    'word': RE_ENGLISH_WORD, 'word0': RE_OPTIONAL_WORD,
    'boldstart': RE_BOLD_START, 'boldend': RE_BOLD_END,
    'boldcharstart': RE_BOLD_CHAR_START, 'boldcharend': RE_BOLD_CHAR_END,
    'italicstart': RE_ITALIC_START, 'italicend': RE_ITALIC_END,
    'italiccharstart': RE_ITALIC_CHAR_START, 'italiccharend': RE_ITALIC_CHAR_END,
    'stylestart': RE_STYLE_START, 'styleend': RE_STYLE_END,
    }

PATTERNS.update({'stylestart': RE_STYLE_START, 'styleend': RE_STYLE_END})
CHARCLASSES = {'w': RE_WORD_CHARCLASS, 'W': RE_BREAK_CHARCLASS, 'b': RE_BREAK_CHARCLASS}
PATTERNS.update(CHARCLASSES)

RE_ACRONYM2 = r'\b(?P<s2>' \
    r'{stylestart}?([a-zA-Z]){styleend}?{word}{styleend}?{b}' \
    r'{stylestart}?([a-zA-Z]){styleend}?{word}{styleend}?' \
    r')[\s]?[\s]?\((?P<a2>\2[-.*_]?[\s]?\3[.]?)\)'.format(**PATTERNS)
RE_ACRONYM3 = r'\b[_*]{0,2}(?P<s3>(\w)[-*\w0-9]{0,16}[ ](\w)[-*\w0-9]{0,16}' \
    r'[ ](\w)[-*\w0-9]{0,16})[_*]{0,2}[ ]\((?P<a3>\6[-.*_ ]{0,2}\7[-.*_ ]{0,2}\8[-.*_ ]{0,2})\)'
RE_ACRONYM4 = r'\b[_*]{0,2}(?P<s4>(\w)[-*\w0-9]{0,16}[ ](\w)[-*\w0-9]{0,16}' \
    r'[ ](\w)[-*\w0-9]{0,16}[ ](\w)[-*\w0-9]{0,16})[_*]{0,2}[ ]' \
    r'\((?P<a4>\11[-.*_ ]{0,2}\12[-.*_ ]{0,2}\13[-.*_ ]{0,2}\14[-.*_ ]{0,2})\)'
RE_ACRONYM5 = r'\b[_*]{0,2}(?P<s5>(\w)[-\w0-9]{0,16}[ ](\w)[-\w0-9]{0,16}' \
    r'[ ](\w)[-*\w0-9]{0,16}[ ](\w)[-*\w0-9]{0,16}[ ](\w)[-*\w0-9]{0,16})' \
    r'[_*]{0,2}[ ]\((?P<a5>\17[-.*_ ]{0,2}\18[-.*_ ]{0,2}\19[-.*_ ]{0,2}\20[-.*_ ]{0,2}\21[-.*_ ]{0,2})\)'
CRE_ACRONYM = re.compile('|'.join((RE_ACRONYM2, RE_ACRONYM3, RE_ACRONYM4, RE_ACRONYM5)), re.IGNORECASE)
