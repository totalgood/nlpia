import os
import requests
# import time

from tqdm import tqdm

# from pugnlp import mkdir_p

USER_HOME = os.path.expanduser("~")
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
DATA_URL = 'http://totalgood.org/static/data'
W2V_FILE = 'GoogleNews-vectors-negative300.bin.gz'
W2V_URL = 'https://www.dropbox.com/s/4bcegydk3pn9067/GoogleNews-vectors-negative300.bin.gz?dl=0'
W2V_PATH = os.path.join(DATA_PATH, W2V_FILE)

with open(os.path.join(DATA_PATH, 'kite.txt')) as f:
    kite_text = f.read()

with open(os.path.join(DATA_PATH, 'kite_history.txt')) as f:
    kite_history = f.read()

harry_docs = ["The faster Harry got to the store, the faster and faster Harry would get home."]
harry_docs += ["Harry is hairy and faster than Jill."]
harry_docs += ["Jill is not as hairy as Harry."]


def no_tqdm(it, total=1):
    return it


def download(names=None, verbose=True):
    names = [names] if isinstance(names, (str, bytes, basestring)) else names
    names = names or ['w2v']
    file_paths = {}
    for name in names:
        name = name.lower().strip()
        if name in ('w2v', 'word2vec'):
            download_file(W2V_URL, 'GoogleNews-vectors-negative300.bin.gz', verbose=verbose)
    return file_paths


def download_file(url, local_file_path=None, chunk_size=1024, verbose=True):
    """Uses stream=True and a reasonable chunk size to be able to download large (GB) files over https"""
    local_file_path = os.path.join(DATA_PATH, url.split('/')[-1]) if local_file_path is None else local_file_path
    if not (local_file_path.startswith(DATA_PATH) or local_file_path[0] in ('/', '~')):
        local_file_path = os.path.join(DATA_PATH, local_file_path)
    if verbose:
        tqdm_prog = tqdm
        print('requesting URL: {}'.format(W2V_URL))
    else:
        tqdm_prog = no_tqdm

    r = requests.get(url, stream=True)
    size = r.headers.get('Content-Length', None)
    loggly.info(r.headers.keys())
    loggly.info('size: {}'.format(size))
    with open(local_file_path, 'wb') as f:
        for chunk in tqdm_prog(r.iter_content(chunk_size=chunk_size),
                               total=size if size is None else int(1 + 2. * size / chunk_size)):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)
    return local_file_path
