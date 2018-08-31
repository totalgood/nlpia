# -*- coding: utf-8 -*-
""" Compiled regular expressions for extracting dates, times, acronyms, etc

>>> CRE_ACRONYM.findall('National Science Foundation (NSF)')
[('', '', '', '', 'National Science Foundation', 'N', 'S', 'F', 'NSF')]
>>> CRE_ACRONYM.findall('National science Foundation (NSF)')
[('', '', '', '', 'National science Foundation', 'N', 's', 'F', 'NSF')]

"""
from nlpia.constants import logging
import re


logger = logging.getLogger(__name__)

RE_ACRONYM2 = r'\b[_*]{0,2}(?P<s2>(\w)[-*\w0-9]{2,16}[ ](\w)[-*\w0-9]{2,16}' \
    r')[_*]{0,2}[ ]\((?P<a2>\2\3)\)'
RE_ACRONYM3 = r'\b[_*]{0,2}(?P<s3>(\w)[-*\w0-9]{2,16}[ ](\w)[-*\w0-9]{2,16}' \
    r'[ ](\w)[-*\w0-9]{2,16})[_*]{0,2}[ ]\((?P<a3>\6\7\8)\)'
RE_ACRONYM4 = r'\b[_*]{0,2}(?P<s4>(\w)[-*\w0-9]{2,16}[ ](\w)[-*\w0-9]{2,16}' \
    r'[ ](\w)[-*\w0-9]{2,16}[ ](\w)[-*\w0-9]{2,16})[_*]{0,2}[ ]\((?P<a4>\11\12\13\14)\)'
RE_ACRONYM5 = r'\b[_*]{0,2}(?P<s5>(\w)[-\w0-9]{2,16}[ ](\w)[-\w0-9]{2,16}' \
    r'[ ](\w)[-*\w0-9]{2,16}[ ](\w)[-*\w0-9]{2,16}[ ](\w)[-*\w0-9]{2,16})' \
    r'[_*]{0,2}[ ]\((?P<a5>\17\18\19\20\21)\)'
CRE_ACRONYM = re.compile('|'.join((RE_ACRONYM2, RE_ACRONYM3, RE_ACRONYM4, RE_ACRONYM5)), re.IGNORECASE)
