#!/usr/bin/env python
""" Web URL maniuplation and downloading/scraping/ParsingError scripts

- Google Drive file download
- Dropbox URL parsing and download
"""
import sys
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from urllib.error import URLError
from lxml.html import fromstring as parse_html

from nlpia.constants import logging

logger = logging.getLogger(__name__)


GOOGLE_DRIVE_PREFIX = 'https://drive.google.com/open?id='

GOOGLE_DRIVEID_FILENAMES = """
1VWkj1oQS2RUhyJXckx3OaDYs5fx2mMCq VGG_ILSVRC2016_SSD_300x300_iter_440000.h5
1LcBPsd9CJbuBw4KiSuE1o1fMA-Pz2Zvw VGG_ilsvrc15_SSD_500x500_iter_480000.h5
1IJWZKmjkcFMlvaz2gYukzFx4d6mH3py5 VGG_coco_SSD_512x512_iter_360000.h5
1vmEF7FUsWfHquXyCqO17UaXOPpRbwsdj VGG_coco_SSD_300x300_iter_400000.h5
121-kCXaOHOkJE_Kf5lKcJvC_5q1fYb_q VGG_VOC0712_SSD_300x300_iter_120000.h5
19NIa0baRCFYT3iRxQkOKCD7CpN6BFO8p VGG_VOC0712_SSD_512x512_iter_120000.h5
1M99knPZ4DpY9tI60iZqxXsAxX2bYWDvZ VGG_VOC0712Plus_SSD_300x300_iter_240000.h5
18nFnqv9fG5Rh_fx6vUtOoQHOLySt4fEx VGG_VOC0712Plus_SSD_512x512_iter_240000.h5
17G1J4zEpFwiOzgBmq886ci4P3YaIz8bY VGG_coco_SSD_300x300.h5
1wGc368WyXSHZOv4iow2tri9LnB0vm9X- VGG_coco_SSD_512x512.h5
1vtNI6kSnv7fkozl7WxyhGyReB6JvDM41 VGG_VOC0712_SSD_300x300_ft_iter_120000.h
14mELuzm0OvXnwjb0mzAiG-Ake9_NP_LQ VGG_VOC0712_SSD_512x512_ft_iter_120000.h5
1fyDDUcIOSjeiP08vl1WCndcFdtboFXua VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5
1a-64b6y6xsQr5puUsHX_wxI1orQDercM VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.h5
"""


def try_parse_url(url):
    """ User urlparse to try to parse URL returning None on exception """
    if len(url.strip()) < 4:
        logger.info('URL too short: {}'.format(url))
        return None
    try:
        parsed_url = urlparse(url)
    except ValueError:
        logger.info('Parse URL ValueError: {}'.format(url))
        return None
    if parsed_url.scheme:
        return parsed_url
    try:
        parsed_url = urlparse('http://' + parsed_url.geturl())
    except ValueError:
        logger.info('Invalid URL for assumed http scheme: urlparse("{}") from "{}" '.format('http://' + parsed_url.geturl(), url))
        return None
    if not parsed_url.scheme:
        logger.info('Unable to guess a scheme for URL: {}'.format(url))
        return None
    return parsed_url


def get_url_filemeta(url):
    """ Request HTML for the page at the URL indicated and return the url, filename, and remote size

    TODO: just add remote_size and basename and filename attributes to the urlparse object
          instead of returning a dict

    >>> sorted(get_url_filemeta('mozilla.com').items())
    [('filename', ''),
     ('hostname', 'mozilla.com'),
     ('path', ''),
     ('remote_size', -1),
     ('url', 'http://mozilla.com'),
     ('username', None)]
    >>> sorted(get_url_filemeta('https://duckduckgo.com/about?q=nlp').items())
    [('filename', 'about'),
     ('hostname', 'duckduckgo.com'),
     ('path', '/about'),
     ('remote_size', -1),
     ('url', 'https://duckduckgo.com/about?q=nlp'),
     ('username', None)]
    >>> 1000 <= int(get_url_filemeta('en.wikipedia.org')['remote_size']) <= 200000
    True
    """
    parsed_url = try_parse_url(url)

    if parsed_url is None:
        return None
    if parsed_url.scheme.startswith('ftp'):
        return get_ftp_filemeta(parsed_url)

    url = parsed_url.geturl()
    try:
        r = requests.get(url, stream=True, allow_redirects=True, timeout=5)
        remote_size = r.headers.get('Content-Length', -1)
        return dict(url=url, hostname=parsed_url.hostname, path=parsed_url.path,
                    username=parsed_url.username, remote_size=remote_size,
                    filename=os.path.basename(parsed_url.path))
    except ConnectionError:
        return None
    except (InvalidURL, InvalidSchema, InvalidHeader, MissingSchema):
        return None
    return None


def get_url_title(url):
    r""" Request HTML for the page at the URL indicated and return it's <title> property

    >>> get_url_title('mozilla.com').strip()
    'Internet for people, not profit\n    — Mozilla'
    """
    parsed_url = try_parse_url(url)
    if parsed_url is None:
        return None
    try:
        r = requests.get(parsed_url.geturl(), stream=False, allow_redirects=True, timeout=5)
        tree = parse_html(r.content)
        title = tree.findtext('.//title')
        return title
    except ConnectionError:
        logging.error('Unable to connect to internet to retrieve URL {}'.format(parsed_url.geturl()))
        logging.error(format_exc())
    except (InvalidURL, InvalidSchema, InvalidHeader, MissingSchema):
        logging.warn('Unable to retrieve URL {}'.format(parsed_url.geturl()))
        logging.error(format_exc())


def get_url_id(url):
    """ Extract the drive ID from a google_drive url """


def get_url_filename(url=None, driveid=None):
    """ Get the filename associated with a google drive driveid or drive.google.com URL

    >>> get_url_filename(driveid='1a-64b6y6xsQr5puUsHX_wxI1orQDercM')
    'VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.h5'
    >>> get_url_filename(url='https://drive.google.com/open?id=' + '1fyDDUcIOSjeiP08vl1WCndcFdtboFXua')
    'VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5'
    >>> get_url_filename(f'{GOOGLE_DRIVE_PREFIX}{driveid}'.format(GOOGLE_DRIVE_PREFIX=GOOGLE_DRIVE_PREFIX, driveid='14mELuzm0OvXnwjb0mzAiG-Ake9_NP_LQ'))
    'VGG_VOC0712_SSD_512x512_ft_iter_120000.h5'
    """
    url = url or 'https://drive.google.com/open?id={}'.format(driveid)
    if url.startswith('https://drive.google.com'):
        filename = get_url_title(url)
        if filename.endswith('Google Drive'):
            filename = filename[:-len('Google Drive')].rstrip().rstrip('-:').rstrip()
        return filename
    logger.warn('Unable to find filename for the URL "{}"'.format(url))
    

def download_file_from_google_drive(driveid, destination=None):
    """ Download script for google drive shared links 

    Thank you @turdus-merula and Andrew Hundt! 
    https://stackoverflow.com/a/39225039/623735
    """
    if '&id=' in driveid:
        # https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs  # dailymail_stories.tgz
        driveid = driveid.split('&id=')[-1]
    if '?id=' in driveid:
        # 'https://drive.google.com/open?id=14mELuzm0OvXnwjb0mzAiG-Ake9_NP_LQ'  # SSD pretrainined keras model
        driveid = driveid.split('?id=')[-1]

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': driveid}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': driveid, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    destination = destination or get_url_filename(driveid=driveid)

    save_response_content(response, destination)    


def main(driveid=None, filename=None):
    if driveid is None:
        if len(sys.argv) < 2:
            print("Usage: python google_drive.py drive_file_id destination_file_path")
            return
        else:
            driveid = sys.argv[1]  # TAKE ID FROM SHAREABLE LINK
        if len(sys.argv) > 2:
            filename = sys.argv[2]  # DESTINATION FILE ON YOUR DISK
        download_file_from_google_drive(driveid, filename)
