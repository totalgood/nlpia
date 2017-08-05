""" Translate documents in some way, like `sed`, only a bit more complex """
import requests
import re

from pugnlp.futil import find_files

from constants import secrets


def minify_urls(filepath, ext='asc', url_regex=None, output_ext='.urls_minified', access_token=None):
    """ Use bitly or similar minifier to shrink all URLs in text files within a folder structure, like the NLPIA manuscript directory

    bitly API: https://dev.bitly.com/links.html

    Args:
      path (str): Directory or file path
      ext (str): File name extension to filter text files by. default='.asc'
      output_ext (str): Extension to append to filenames of altered files default='' (in-place replacement of URLs)

    FIXME: NotImplementedError! Untested!
    """
    access_token = access_token or secrets.bitly.access_token
    output_ext = output_ext or ''
    url_regex = re.compile(url_regex) if isinstance(url_regex, str) else url_regex
    filemetas = []
    for filemeta in find_files(filepath, ext=ext):
        filemetas += [filemeta]
        altered_text = ''
        with open(filemeta['path'], 'rt') as fin:
            text = fin.read()
        end = 0
        for match in url_regex.finditer(text):
            url = match.group()
            start = match.start()
            altered_text += text[:start]
            resp = requests.get('https://api-ssl.bitly.com/v3/shorten?access_token={}&longUrl={}'.format(access_token, url))
            js = resp.json()
            short_url = js['shortUrl']
            altered_text += short_url
            end = start + len(url)
        altered_text += text[end:]
        with open(filemeta['path'] + (output_ext or ''), 'wt') as fout:
            fout.write(altered_text)
    return altered_text


def segment_sentences(filepath, ext='asc'):
    """ Insert and delete newlines in a text document to produce once sentence or heading per line.

    Lines are labeled with their classification as "sentence" or "phrase" (e.g title or heading)

    1. process each line with an agressive sentence segmenter, like DetectorMorse
    2. process our manuscript to create a complete-sentence and heading training set normalized/simplified syntax net tree is the input feature set
       common words and N-grams inserted with their label as additional feature
    3. process a training set with a grammar checker and sentax next to bootstrap a "complete sentence" labeler.
    4. process each 1-3 line window (breaking on empty lines) with syntax net to label them
    5. label each 1-3-line window of lines as "complete sentence, partial sentence/phrase, or multi-sentence"
    """
    for filemeta in find_files(filepath, ext=ext):
        altered_text = ''
        with open(filemeta['path'], 'rt') as fin:
            for line in fin:
                altered_text += line
