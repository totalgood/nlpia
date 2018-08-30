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

RE_ACRONYM2 = r'((\w)[\w0-9]{2,16}[ ](\w)[\w0-9]{2,16})[ ]\((\2\3)\)'
RE_ACRONYM3 = r'((\w)[\w0-9]{2,16}[ ](\w)[\w0-9]{2,16}[ ](\w)[\w0-9]{2,16})[ ]\((\6\7\8)\)'
CRE_ACRONYM = re.compile(RE_ACRONYM2 + '|' + RE_ACRONYM3, re.IGNORECASE)
