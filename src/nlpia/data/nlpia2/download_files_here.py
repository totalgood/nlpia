# download_files.py

URLS = [
    'https://en.wikiquote.org/wiki/The_Book_Thief#Quotes'
]


import requests
from bs4 import BeautifulSoup


scrape_html_section(url='https://en.wikiquote.org/wiki/The_Book_Thief#Quotes'):
    resp = requests.get(url)
    anchors = url.split('#')
    if len(anchors) == 2:
        section = anchors[-1]
    else:
        section = ''
    soup = BeautifulSoup(resp.text)
    lines = resp.text.splitlines()
    for s in soup.text.splitlines():
        if 'Quotes' in s:
            print(s)
