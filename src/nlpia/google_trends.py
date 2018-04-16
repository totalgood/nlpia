""" Deprecated Google Trends API """
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import (bytes, dict, int, list, object, range, str,  # noqa
    ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict,

import io

import mechanicalsoup as mechanize
import pandas as pd


def get_google_trends(un, pw):
    br = mechanize.StatefulBrowser()

    br.addheaders = [
        ('User-agent',
         'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1'
         )]
    response = br.open('https://accounts.google.com/ServiceLogin?hl=en&continue=https://www.google.com/')
    response.soup.find_all("input", {"id": "Email"})[0]['value'] = un
    response.soup.find_all("input", {"id": "Passwd-hidden"})[0]['value'] = pw
    form = response.soup.select("form")[0]
    print(form)
    form_response = br.open(form.click())
    print(form_response)

    # google no longer provides tabular trends data:
    table = br.open("http://www.google.com/trends/trendsReport?q=SearchTerm&export=1")
    return pd.read_csv(io.StringIO(table.text))
