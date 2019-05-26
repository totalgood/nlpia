# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import (bytes, dict, int, list, object, range, str,  # noqa
    ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
from future import standard_library
from past.builtins import basestring
standard_library.install_aliases()  # noqa

import itertools
import json  # noqa (unused)
import os
import re
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError  # MissingSchema
import tempfile  # noqa (unused)
from urllib.parse import urlparse
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import pandas as pd

from pugnlp.futil import find_files

from nlpia.futil import find_filepath, ensure_open, read_json, read_csv, read_text
from nlpia.constants import logging, DATA_PATH, BIGDATA_PATH, BOOK_PATH  # noqa
from nlpia.constants import HTML_TAGS, EOL
from nlpia.constants import UTF8_TO_ASCII, UTF8_TO_MULTIASCII
from nlpia.web import try_parse_url, looks_like_url, http_status_code


np = pd.np
logger = logging.getLogger(__name__)


def format_hex(i, num_bytes=4, prefix='0x'):
    """ Format hexidecimal string from decimal integer value

    >>> format_hex(42, num_bytes=8, prefix=None)
    '0000002a'
    >>> format_hex(23)
    '0x0017'
    """
    prefix = str(prefix or '')
    i = int(i or 0)
    return prefix + '{0:0{1}x}'.format(i, num_bytes)


NAME_ACCENT = {
    'CIRCUMFLEX': '^',
    'DIAERESIS': ':',
    'TILDE': '~',
    'GRAVE': '`',
    'ACCUTE': "'",
    'GRAVE ACCENT': '`',
    'ACCUTE ACCENT': "'",
    'GRAVE ACCENT ABOVE': '`',
    'ACCUTE ACCENT ABOVE': "'",
}

NAME_ASCII = {
    'ETH': 'D',
    'AE': 'E',
    'SHARP S': 'B',
    'DOTLESS I': 'I',
}


def prepend_http(url):
    """ Ensure there's a scheme specified at the beginning of a url, defaulting to http://

    >>> prepend_http('duckduckgo.com')
    'http://duckduckgo.com'
    """
    url = url.lstrip()
    if not urlparse(url).scheme:
        return 'http://' + url
    return url


def is_up_url(url, allow_redirects=False, timeout=5):
    r""" Check URL to see if it is a valid web page, return the redirected location if it is

    Returns:
      None if ConnectionError
      False if url is invalid (any HTTP error code)
      cleaned up URL (following redirects and possibly adding HTTP schema "http://")

    >>> is_up_url("duckduckgo.com")  # a more private, less manipulative search engine
    'https://duckduckgo.com/'
    >>> urlisup = is_up_url("totalgood.org")
    >>> not urlisup or str(urlisup).startswith('http')
    True
    >>> urlisup = is_up_url("wikipedia.org")
    >>> str(urlisup).startswith('http')
    True
    >>> 'wikipedia.org' in str(urlisup)
    True
    >>> bool(is_up_url('8158989668202919656'))
    False
    >>> is_up_url('invalidurlwithoutadomain')
    False
    """
    if not isinstance(url, basestring) or '.' not in url:
        return False
    normalized_url = prepend_http(url)
    session = requests.Session()
    session.mount(url, HTTPAdapter(max_retries=2))
    try:
        resp = session.get(normalized_url, allow_redirects=allow_redirects, timeout=timeout)
    except ConnectionError:
        return None
    except:  # noqa (bare except)
        return None
    if resp.status_code in (301, 302, 307) or resp.headers.get('location', None):
        return resp.headers.get('location', None)  # return redirected URL
    elif 100 <= resp.status_code < 400:
        return normalized_url  # return the original URL that was requested/visited
    else:
        return False


def get_markdown_levels(lines, levels=set((0, 1, 2, 3, 4, 5, 6))):
    r""" Return a list of 2-tuples with a level integer for the heading levels

    >>> get_markdown_levels('paragraph \n##bad\n# hello\n  ### world\n')
    [(0, 'paragraph '), (2, 'bad'), (0, '# hello'), (3, 'world')]
    >>> get_markdown_levels('- bullet \n##bad\n# hello\n  ### world\n')
    [(0, '- bullet '), (2, 'bad'), (0, '# hello'), (3, 'world')]
    >>> get_markdown_levels('- bullet \n##bad\n# hello\n  ### world\n', 2)
    [(2, 'bad')]
    >>> get_markdown_levels('- bullet \n##bad\n# hello\n  ### world\n', 1)
    []

    """
    if isinstance(levels, (int, float, basestring, str, bytes)):
        levels = [float(levels)]
    levels = set([int(i) for i in levels])
    if isinstance(lines, basestring):
        lines = lines.splitlines()
    level_lines = []
    for line in lines:
        level_line = None
        if 0 in levels:
            level_line = (0, line)
        lstripped = line.lstrip()
        for i in range(6, 1, -1):
            if lstripped.startswith('#' * i):
                level_line = (i, lstripped[i:].lstrip())
                break
        if level_line and level_line[0] in levels:
            level_lines.append(level_line)
    return level_lines


def read_http_status_codes(filename='HTTP_1.1  Status Code Definitions.html'):
    r""" Parse the HTTP documentation HTML page in filename

    Return:
        code_dict: {200: "OK", ...}

    >>> fn = 'HTTP_1.1  Status Code Definitions.html'
    >>> code_dict = read_http_status_codes(fn)
    >>> code_dict
    {'100': 'Continue',
    100: 'Continue',
    '101': 'Switching Protocols',
    101: 'Switching Protocols',
    '200': 'OK',
    200: 'OK',...
    >>> json.dump(code_dict, open(os.path.join(DATA_PATH, fn + '.json'), 'wt'), indent=2)
    """
    lines = read_text(filename)
    level_lines = get_markdown_levels(lines, 3)
    code_dict = {}
    for level, line in level_lines:
        code, name = (re.findall(r'\s(\d\d\d)[\W]+([-\w\s]*)', line) or [[0, '']])[0]
        if 1000 > int(code) >= 100:
            code_dict[code] = name
            code_dict[int(code)] = name
    return code_dict


def iter_lines(url_or_text, ext=None, mode='rt'):
    r""" Return an iterator over the lines of a file or URI response.

    >>> len(list(iter_lines('cats_and_dogs.txt')))
    263
    >>> len(list(iter_lines(list('abcdefgh'))))
    8
    >>> len(list(iter_lines('abc\n def\n gh\n')))
    3
    >>> len(list(iter_lines('abc\n def\n gh')))
    3
    >>> 20000 > len(list(iter_lines(BOOK_PATH))) > 200
    True
    """
    if url_or_text is None or not url_or_text:
        return []
        # url_or_text = 'https://www.fileformat.info/info/charset/UTF-8/list.htm'
    elif isinstance(url_or_text, (str, bytes, basestring)):
        if '\n' in url_or_text or '\r' in url_or_text:
            return StringIO(url_or_text)
        elif os.path.isfile(os.path.join(DATA_PATH, url_or_text)):
            return open(os.path.join(DATA_PATH, url_or_text), mode=mode)
        elif os.path.isfile(url_or_text):
            return open(os.path.join(url_or_text), mode=mode)
        if os.path.isdir(url_or_text):
            filepaths = [filemeta['path'] for filemeta in find_files(url_or_text, ext=ext)]
            return itertools.chain.from_iterable(map(open, filepaths))
        url = looks_like_url(url_or_text)
        if url:
            for i in range(3):
                return requests_get(url, stream=True, allow_redirects=True, timeout=5)
        else:
            return StringIO(url_or_text)
    elif isinstance(url_or_text, (list, tuple)):
        # FIXME: make this lazy with chain and map so it doesn't gobble up RAM
        text = ''
        for s in url_or_text:
            text += '\n'.join(list(iter_lines(s, ext=ext, mode=mode))) + '\n'
        return iter_lines(text)


def parse_utf_html(url=os.path.join(DATA_PATH, 'utf8_table.html')):
    """ Parse HTML table UTF8 char descriptions returning DataFrame with `ascii` and `mutliascii` """
    utf = pd.read_html(url)
    utf = [df for df in utf if len(df) > 1023 and len(df.columns) > 2][0]
    utf = utf.iloc[:1024] if len(utf) == 1025 else utf
    utf.columns = 'char name hex'.split()
    utf.name = utf.name.str.replace('<control>', 'CONTTROL CHARACTER')
    multiascii = [' '] * len(utf)
    asc = [' '] * len(utf)
    rows = []
    for i, name in enumerate(utf.name):
        if i < 128 and str.isprintable(chr(i)):
            asc[i] = chr(i)
        else:
            asc[i] = ' '
        big = re.findall(r'CAPITAL\ LETTER\ ([a-z0-9A-Z ]+$)', name)
        small = re.findall(r'SMALL\ LETTER\ ([a-z0-9A-Z ]+$)', name)
        pattern = r'(?P<description>' \
                      r'(?P<lang>LATIN|GREEK|COPTIC|CYRILLIC)?[\s]*' \
                      r'(?P<case>CAPITAL|SMALL)?[\s]*' \
                      r'(?P<length>CHARACTER|LETTER)?[\s]*' \
                      r'(?P<ukrainian>BYELORUSSIAN-UKRAINIAN)?[\s]*' \
                      r'(?P<name>[-_><a-z0-9A-Z\s ]+)[\s]*' \
                      r'\(?(?P<code_point>U\+[- a-fA-F0-9]{4,8})?\)?)[\s]*'  # noqa
        match = re.match(pattern, name)
        gd = match.groupdict()
        gd['char'] = chr(i)
        gd['suffix'] = None
        gd['wordwith'] = None

        withprefix = re.match(r'(?P<prefix>DOTLESS|TURNED|SMALL)(?P<name>.*)' +
                              r'(?P<wordwith>WITH|SUPERSCRIPT|SUBSCRIPT|DIGRAPH)\s+(?P<suffix>[-_><a-z0-9A-Z\s ]+)',
                              gd['name'])
        if withprefix:
            gd.update(withprefix.groupdict())

        withsuffix = re.match(r'(?P<name>.*)(?P<wordwith>WITH|SUPERSCRIPT|SUBSCRIPT|DIGRAPH)\s+' +
                              r'(?P<suffix>[-_><a-z0-9A-Z\s ]+)',
                              gd['name'])
        if withsuffix:
            gd.update(withsuffix.groupdict())

        gd['code_point'] = gd['code_point'] or format_hex(i, num_bytes=4, prefix='U+').upper()
        if i < 128:
            gd['ascii'] = chr(i)
        else:
            multiascii = gd['name']
            if gd['suffix'] and gd['wordwith']:
                multiascii = NAME_ACCENT.get(gd['suffix'], "'")
            else:
                if big:
                    m = big[0]
                    multiascii[i] = m
                    if len(m) == 1:
                        asc[i] = m
                elif small:
                    multiascii[i] = small[0].lower()
                    if len(multiascii[i]) == 1:
                        asc[i] = small[0].lower()
        rows.append(gd)
    df = pd.DataFrame(rows)
    df.multiascii = df.multiascii.str.strip()
    df['ascii'] = df['ascii'].str.strip()
    df.name = df.name.str.strip()
    return df


def clean_csvs(dialogpath=None):
    """ Translate non-ASCII characters to spaces or equivalent ASCII characters """
    dialogdir = os.dirname(dialogpath) if os.path.isfile(dialogpath) else dialogpath
    filenames = [dialogpath.split(os.path.sep)[-1]] if os.path.isfile(dialogpath) else os.listdir(dialogpath)
    for filename in filenames:
        filepath = os.path.join(dialogdir, filename)
        df = clean_df(filepath)
        df.to_csv(filepath, header=None)
    return filenames


def unicode2ascii(text, expand=True):
    r""" Translate UTF8 characters to ASCII

    >> unicode2ascii("żółw")
    zozw

    utf8_letters =  'ą ę ć ź ż ó ł ń ś “ ” ’'.split()
    ascii_letters = 'a e c z z o l n s " " \''
    """
    translate = UTF8_TO_ASCII if not expand else UTF8_TO_MULTIASCII
    output = ''
    for c in text:
        if not c or ord(c) < 128:
            output += c
        else:
            output += translate[c] if c in translate else ' '
    return output.strip()


def clean_df(df, header=None, **read_csv_kwargs):
    """ Convert UTF8 characters in a CSV file or dataframe into ASCII

    Args:
      df (DataFrame or str): DataFrame or path or url to CSV
    """
    df = read_csv(df, header=header, **read_csv_kwargs)
    df = df.fillna(' ')
    for col in df.columns:
        df[col] = df[col].apply(unicode2ascii)
    return df

