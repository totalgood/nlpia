""" Utilities for transforming the syntax/style/grammar of a document, usually asciidoc or markdown 

Instantiates Objects derived from the `_sre.SRE_Pattern` class (compiled regular expressions) so they work with regex.sub()
"""
import logging
from nlpia.regexes import Pattern, RE_HYPERLINK

# from nlpia.constants import DATA_PATH
logger = logging.getLogger(__name__)


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

    def replace(self, text, to_template, from_template=None):
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
        matches = self.finditer(text)
        newdoc = text
        for m in matches:
            # this outer m.captures() loop is overkill:
            #   overlapping pattern matches probably won't match after the first replace
            for i, captured_str in enumerate(m.captures()):
                if from_template:
                    rendered_from_template = from_template.format(
                        **dict((k, v[i]) for k, v in m.capturesdict().items())) 
                else:
                    rendered_from_template = captured_str
                # TODO: render numbered references like r'\1' before rendering named references
                #    or do them together in one `.format(**kwargs)` after translating \1 to {1} and groupsdict().update({1: ...})
                rendered_to_template = to_template.format(**m.groupdict())
                newdoc = newdoc.replace(rendered_from_template, rendered_to_template)
        return newdoc

    def translate(self, text, to_template='{name} ({url})', from_template=None):
        """ Translate hyperinks into printable book style for Manning Publishing

        >>> translator = HyperlinkStyleCorrector()
        >>> adoc = 'See http://totalgood.com[Total Good] about that.'
        >>> translator.translate(adoc)
        'See Total Good (http://totalgood.com) about that.'
        """
        return self.replace(text, to_template=to_template, from_template=None)

