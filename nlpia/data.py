import os
import requests
import time

from tqdm import tqdm

# from pugnlp import mkdir_p

USER_HOME = os.path.expanduser("~")
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
DATA_URL = 'http://totalgood.org/static/data'
W2V_FILE = 'GoogleNews-vectors-negative300.bin.gz'
W2V_URL = DATA_URL.rstrip('/') + '/' + W2V_FILE
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


def download(name=None, verbose=True):
    name = (name or '').lower()
    file_paths = {}
    if verbose:
        tqdm_prog = tqdm
    else:
        tqdm_prog = no_tqdm
    if name in ('', 'w2v', 'word2vec'):
        if verbose:
            print('requesting URL: {}'.format(W2V_URL))
        req = requests.get(W2V_URL)
        if verbose:
            print('response URL: {}'.format(req))
        with open(W2V_FILE, 'wb') as f:
            for chunk in tqdm_prog(req.iter_content(100000), total=int(f.size / 100000. + 1)):
                f.write(chunk)
                time.sleep(.001)
            file_paths['w2v'] = f.name
    return file_paths
