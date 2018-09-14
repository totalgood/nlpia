""" Utilities for transforming the syntax/style/grammar of a document, usually asciidoc or markdown 

Instantiates Objects derived from the `_sre.SRE_Pattern` class (compiled regular expressions) so they work with regex.sub()
"""
import logging
import re
import regex

# from nlpia.constants import DATA_PATH
logger = logging.getLogger(__name__)


class Pattern:
    """ Container for _regex.Pattern object augmented with Irregular matching rules """

    def __init__(self, pattern):
        self._compiled_pattern = regex.compile(pattern)
        for name in dir(self._compiled_pattern):
            if name in ('__class__', '__init__'):
                continue
            attr = getattr(self._compiled_pattern, name)
            try:
                setattr(self, name, attr)
                logger.debug('{}.{}.Pattern successfully "inherited" `_regex.Pattern.{}{}`'.format(
                    __package__, __name__, name, '()' if callable(attr) else ''))
            except:
                logger.warning('Unable to "inherit" `_regex.Pattern.{}{}`'.format(
                    name, '()' if callable(attr) else ''))


class REPattern:
    """ Container for _regex.Pattern object augmented with Irregular matching rules """

    def __init__(self, pattern):
        self._compiled_pattern = re.compile(pattern)
        for name in dir(self._compiled_pattern):
            if name in ('__class__', '__init__'):
                continue
            attr = getattr(self._compiled_pattern, name)
            try:
                setattr(self, name, attr)
                logger.debug('{}.{}.{} successfully "inherited" `_regex.Pattern.{}{}`'.format(
                    __package__, __name__, self.__class__, name, '()' if callable(attr) else ''))
            except:
                logger.warning('Unable to "inherit" `_regex.Pattern.{}{}`'.format(
                    name, '()' if callable(attr) else ''))
