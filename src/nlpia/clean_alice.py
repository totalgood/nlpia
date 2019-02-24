""" AIML Loader that can load zipped AIML2.0 XML files with an AIML1.0 parser in python 3

TODO:
  fix doctests

>>> from nlpia.loaders import get_data

>> alice_path = get_data('alice')
>> bot = create_brain(alice_path)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
Loading ...
>> len(bot._brain._root.keys())
3445
>> bot._brain._root['HI']
{'EVERYBODY': {3: {1: {4: {1: {2: ['template', {}, ...

>> bot.respond("Hi how are you?")
'Hi there!. I am fine, thank you.'
>> bot.respond("hi how are you?")
"Hi there!. I'm doing fine thanks how are you?"
>> bot.respond("hi how are you?")
'Hi there!. I am doing very well. How are you  ?'
>> bot.respond("hi how are you?")
'Hi there!. My logic and cognitive functions are normal.'
>> bot.respond("how are you?")
'My logic and cognitive functions are normal.'
>> bot.respond("how are you?")
'I am functioning within normal parameters.'
>> bot.respond("how are you?")
'My logic and cognitive functions are normal.'
>> bot.respond("how are you?")
'I am functioning within normal parameters.'
>> bot.respond("how are you?")
'I am doing very well. How are you  ?'
"""
import os
import zipfile
from traceback import format_exc

from nlpia.constants import logging
logger = logging.getLogger(__name__)

try:
    from aiml_bot import Bot
    from aiml_bot.aiml_parser import AimlParserError
except:
    class Bot:
        pass
    class AimlParserError:
        pass
    logger.error('Unable to import aiml_bot.aiml_parser and aiml_bot.Bot, so nlpia will not be able to parse AIML files.')

from nlpia.constants import logging
from nlpia.constants import BIGDATA_PATH
from nlpia.futil import find_data_path

logger = logging.getLogger(__name__)


def concatenate_aiml(path='aiml-en-us-foundation-alice.v1-9.zip', outfile='aiml-en-us-foundation-alice.v1-9.aiml'):
    """Strip trailing </aiml> tag and concatenate all valid AIML files found in the ZIP."""
    path = find_data_path(path) or path

    zf = zipfile.ZipFile(path)
    for name in zf.namelist():
        if not name.lower().endswith('.aiml'):
            continue
        with zf.open(name) as fin:
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
                    pass

            if happyending != (i, line):
                print('Invalid AIML format: {}\nLast line (line number {}) was: {}\nexpected "</aiml>"'.format(
                    name, i, line))


def extract_aiml(path='aiml-en-us-foundation-alice.v1-9'):
    """ Extract an aiml.zip file if it hasn't been already and return a list of aiml file paths """
    path = find_data_path(path) or path
    if os.path.isdir(path):
        paths = os.listdir(path)
        paths = [os.path.join(path, p) for p in paths]
    else:
        zf = zipfile.ZipFile(path)
        paths = []
        for name in zf.namelist():
            if '.hg/' in name:
                continue
            paths.append(zf.extract(name, path=BIGDATA_PATH))
    return paths


def create_brain(path='aiml-en-us-foundation-alice.v1-9.zip'):
    """ Create an aiml_bot.Bot brain from an AIML zip file or directory of AIML files """
    path = find_data_path(path) or path

    bot = Bot()
    num_templates = bot._brain.template_count
    paths = extract_aiml(path=path)
    for path in paths:
        if not path.lower().endswith('.aiml'):
            continue
        try:
            bot.learn(path)
        except AimlParserError:
            logger.error(format_exc())
            logger.warning('AIML Parse Error: {}'.format(path))
        num_templates = bot._brain.template_count - num_templates
        logger.info('Loaded {} trigger-response pairs.\n'.format(num_templates))
    print('Loaded {} trigger-response pairs from {} AIML files.'.format(bot._brain.template_count, len(paths)))
    return bot
