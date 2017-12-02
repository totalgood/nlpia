import zipfile
from nlpia.constant import DATA_PATH
from aiml_bot.aiml_parser import Bot


def concatenate_aiml(path='aiml-en-us-foundation-alice.v1-9.zip', outfile='aiml-en-us-foundation-alice.v1-9.aiml')
    """Strip trailing </aiml> tag and concatenate all valid AIML files found in the ZIP."""
    if not os.path.isfile(path):
        path = os.path.join(DATA_PATH, path)

    zf = zipfile.ZipFile(path)
    for name in zf.namelist():
        with zf.open(name) as fin:
            # print(name)
            happyending = '#!*@!!BAD'
            for i, line in enumerate(fin):
                try:
                    line = line.decode('utf-8').strip()
                except UnicodeDecodeError:
                    line = line.decode('ISO-8859-1').strip()
                if line.lower().startswith('</aiml>') or line.lower().endswith('</aiml>'):
                    happyending = (i, line)
                    break
                else:

            if happyending != (i, line):
                print('Invalid AIML format: {}\nLast line (line number {}) was: {}\nexpected "</aiml>"'.format(
                    name, i, line)


def extract_aiml(path='aiml-en-us-foundation-alice.v1-9.zip'):
    if not os.path.isfile(path):
        path=os.path.join(DATA_PATH, path)

    zf=zipfile.ZipFile(path)
    for name in f.namelist():
        if '.hg/' in name:
            continue
        f.extract(name, path=DATA_PATH)


def create_brain(path='aiml-en-us-foundation-alice.v1-9.zip'):
    if not os.path.isfile(path):
        path=os.path.join(DATA_PATH, path)

    bot=Bot()
    extract_aiml()
