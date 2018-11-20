""" Utilities for transforming the syntax/style/grammar of a document, usually asciidoc or markdown 

Instantiates Objects derived from the `_sre.SRE_Pattern` class (compiled regular expressions) so they work with regex.sub()
"""
import logging
# Simport regex
from copy import copy

from nlpia.regexes import Pattern, RE_HYPERLINK

# from nlpia.constants import DATA_PATH
logger = logging.getLogger(__name__)


def looks_like_name(s):
    if len(s) < 3:
        return None
    if ' ' in s or '.' not in s or '/' not in s:
        return s


class Matcher(Pattern):
    """ Pattern with additional .ismatch() that returns bool(re.match())

    ismatch is desicned to be overriden by a custom function that returns a bool

    >>> chars = list('Hello World!')
    >>> m = Matcher('[a-z]')
    >>> [m.ismatch(s) for s in chars]
    [False, True, True, True, True, False, False, True, True, True, True, False]
    >>> m = Matcher('^[A-Z][a-z]+$')

    >>> tokens = 'A BIG Hello World to You!'.split()
    >>> m = Matcher(lambda s: len(s) <= 3)
    >>> [m.ismatch(s) for s in tokens]
    [True, True, False, False, True, False]
    >>> m = Matcher(None)
    >>> [m.ismatch(s) for s in tokens]
    [True, True, True, True, True, True]
    """

    def __init__(self, pattern='', ismatchfun=None):
        if callable(pattern):
            ismatchfun = pattern
            pattern = ''
        self._ismatchfun = ismatchfun or self._return_true
        super().__init__(pattern or '')

    def _return_true(self, *args, **kwargs):
        return True

    def _return_false(self, *args, **kwargs):
        return False

    def _ismatchfun(self, s):
        return self._return_true()

    def ismatch(self, s):
        """ like compiled_re.match() but returns True or False """
        if self._ismatchfun(s) and self._compiled_pattern.match(s):
            return True
        return False

    def match(self, s, *args, **kwargs):
        if self.ismatchfun(s):
            return super().match(s, *args, **kwargs)


class Filter(Matcher):
    """ Pattern with additional .ismatch() and .filter() methods

    >>> chars = list('Hello World!')
    >>> m = Filter('[a-z]')
    >>> [m.filter(c) for c in chars]
    ['', 'e', 'l', 'l', 'o', '', '', 'o', 'r', 'l', 'd', '']
    >>> m = Filter('^[A-Z][a-z]+$')
    >>> tokens = 'A BIG Hello World to You!'.split()
    >>> [m.filter(s) for s in tokens]
    ['', '', 'Hello', 'World', '', '']
    >>> m = Filter(None)
    >>> [m.filter(s) for s in tokens] 
    ['A', 'BIG', 'Hello', 'World', 'to', 'You!']
    """

    def filter(self, s):
        """ like compiled_re.match() but returns the entire string if it matches or empty string otherwise """
        if self.ismatch(s):
            return s
        return ''


class Translator(Pattern):
    r""" A pattern for translating a diff file into a more human (non-programmer) readable form

    This is the start of a translator demo that turns diff patch files into human-readable email.
    >> from nlpia.loaders import get_data
    >> difftxt = get_data('forum_user_557658.patch')

    >>> tran = Translator()
    """

    def __init__(self, pattern=r'^\-(?P<text>.*)', template='      was: {text}'):
        super().__init__(pattern=pattern)

    def replace(self, text, to_template='{name} ({url})', from_template=None,
                name_matcher=Matcher(looks_like_name), url_matcher=Matcher(r'.*[^:]+$')):
        """ Replace all occurrences of rendered from_template in text with `template` rendered from each match.groupdict()

        TODO: from_template 

        >>> translator = HyperlinkStyleCorrector()
        >>> adoc = 'See http://totalgood.com[Total Good] about that.'
        >>> translator.replace(adoc, '{scheme_type}s://', '{scheme}://')
        'See http://totalgood.com[Total Good] about that.'
        >>> adoc = "Nada here:// Only a .com & no (parens.symbol) or http/[hyperlinks] or anything!"
        >>> translator.translate(adoc)
        'Nada here:// Only a .com & no (parens.symbol) or http/[hyperlinks] or anything!'
        >>> adoc = "Two http://what.com[WAT] with https://another.com/api?q=1&a=2[longer url]."
        >>> translator.translate(adoc)
        'Two WAT (http://what.com) with longer url (https://another.com/api?q=1&a=2).'
        """
        self.name_matcher = name_matcher or Matcher()
        self.url_matcher = url_matcher or Matcher()
        matches = self.finditer(text)
        newdoc = copy(text)
        logger.debug('before translate: {}'.format(newdoc))
        for m in matches:
            # this outer m.captures() loop is overkill:
            #   overlapping pattern matches probably won't match after the first replace
            logger.debug('match: {}'.format(m))
            logger.debug('match.captures(): {}'.format(m.captures()))
            for i, captured_str in enumerate(m.captures()):
                captureddict = {'name': None, 'scheme': None, 'url': None}
                for k, v in m.capturesdict().items():
                    if len(v) > i:
                        captureddict[k] = v[i]
                    else:
                        captureddict[k] = None
                        logger.warning('Overlapping captured matches were mishandled: {}'.format(m.capturesdict()))
                # need to check for optional args:
                name = captureddict.get('name', None)
                url = captureddict.get('url', None)
                scheme = captureddict.get('scheme', None)
                if (not scheme or not name or not self.name_matcher.ismatch(name) or 
                        not url or not self.url_matcher.ismatch(url)):
                    continue
                if from_template:
                    rendered_from_template = from_template.format(**captureddict)
                else:
                    rendered_from_template = captured_str
                # TODO: render numbered references like r'\1' before rendering named references
                #    or do them together in one `.format(**kwargs)` after translating \1 to {1} and groupsdict().update({1: ...})
                rendered_to_template = to_template.format(**m.groupdict())
                newdoc = newdoc.replace(rendered_from_template, rendered_to_template)
        return newdoc


class HyperlinkStyleCorrector(Pattern):
    """ A pattern for matching asciidoc hyperlinks for transforming them to print-book version (Manning Style)

    >>> adoc = 'See http://totalgood.com[Total Good] about that.'
    >>> translator = HyperlinkStyleCorrector()
    >>> matches = list(translator.finditer(adoc))
    >>> m = matches[0]
    >>> m
    <regex.Match object; span=(4, 36), match='http://totalgood.com[Total Good]'>
    >>> for m in matches:
    ...     newdoc = adoc.replace(
    ...         '{scheme}'.format(**m.groupdict()),
    ...         ''.format(**m.groupdict()))
    >>> newdoc
    'See totalgood.com[Total Good] about that.'
    >>> translator.replace(adoc, '{scheme}', '{scheme_type}s://')
    'See http://totalgood.com[Total Good] about that.'
    """

    def __init__(self, pattern=RE_HYPERLINK):
        super().__init__(pattern=pattern)

    def name_matcher(s):
        return s 

    def replace(self, text, to_template='{name} ({url})', from_template=None,
                name_matcher=Matcher(looks_like_name), url_matcher=Matcher(r'.*[^:]+$')):
        """ Replace all occurrences of rendered from_template in text with `template` rendered from each match.groupdict()

        TODO: from_template 

        >>> translator = HyperlinkStyleCorrector()
        >>> adoc = 'See http://totalgood.com[Total Good] about that.'
        >>> translator.replace(adoc, '{scheme_type}s://', '{scheme}')
        'See https://totalgood.com[Total Good] about that.'
        >>> adoc = "Nada here:// Only a .com & no (parens.symbol) or http/[hyperlinks] or anything!"
        >>> translator.translate(adoc)
        'Nada here:// Only a .com & no (parens.symbol) or http/[hyperlinks] or anything!'
        >>> adoc = "Two http://what.com[WAT] with https://another.com/api?q=1&a=2[longer url]."
        >>> translator.translate(adoc)
        'Two WAT (http://what.com) with longer url (https://another.com/api?q=1&a=2).'
        """
        self.name_matcher = name_matcher or Matcher()
        self.url_matcher = url_matcher or Matcher()
        matches = self.finditer(text)
        newdoc = copy(text)
        logger.debug('before translate: {}'.format(newdoc))
        for m in matches:
            # this outer m.captures() loop is overkill:
            #   overlapping pattern matches probably won't match after the first replace
            logger.debug('match: {}'.format(m))
            logger.debug('match.captures(): {}'.format(m.captures()))
            for i, captured_str in enumerate(m.captures()):
                captureddict = {'name': None, 'scheme': None, 'url': None}
                for k, v in m.capturesdict().items():
                    if len(v) > i:
                        captureddict[k] = v[i]
                    else:
                        captureddict[k] = None
                        logger.warning('Overlapping captured matches were mishandled: {}'.format(m.capturesdict()))
                # need to check for optional args:
                name = captureddict.get('name', None)
                url = captureddict.get('url', None)
                scheme = captureddict.get('scheme', None)
                if (not scheme or not name or not self.name_matcher.ismatch(name) or 
                        not url or not self.url_matcher.ismatch(url)):
                    continue
                if from_template:
                    rendered_from_template = from_template.format(**captureddict)
                else:
                    rendered_from_template = captured_str
                # TODO: render numbered references like r'\1' before rendering named references
                #    or do them together in one `.format(**kwargs)` after translating \1 to {1} and groupsdict().update({1: ...})
                rendered_to_template = to_template.format(**m.groupdict())
                newdoc = newdoc.replace(rendered_from_template, rendered_to_template)
        return newdoc

    def translate(self, text, to_template='{name} ({url})', from_template=None, name_matcher=None, url_matcher=None):
        """ Translate hyperinks into printable book style for Manning Publishing

        >>> translator = HyperlinkStyleCorrector()
        >>> adoc = 'See http://totalgood.com[Total Good] about that.'
        >>> translator.translate(adoc)
        'See Total Good (http://totalgood.com) about that.'
        """
        return self.replace(text, to_template=to_template, from_template=from_template,
                            name_matcher=name_matcher, url_matcher=url_matcher)

