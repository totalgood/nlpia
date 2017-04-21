from __future__ import print_function, unicode_literals, division, absolute_import
# from builtins import int, round, str
from future import standard_library
standard_library.install_aliases()  # NOQA

import io

import mechanize
import cookielib
import pandas as pd

from secrets import accounts_google_pw, accounts_google_un


br = mechanize.Browser()
cj = cookielib.LWPCookieJar()
br.set_cookiejar(cj)

br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
response = br.open('https://accounts.google.com/ServiceLogin?hl=en&continue=https://www.google.com/')
forms = mechanize.ParseResponse(response)
form = forms[0]
form['Email'] = accounts_google_un
form['Passwd'] = accounts_google_pw

response = br.open(form.click())

fstream = br.open("http://www.google.com/trends/trendsReport?q=SearchTerm&export=1")
df = pd.read_csv(io.StringIO(fstream.read()))
