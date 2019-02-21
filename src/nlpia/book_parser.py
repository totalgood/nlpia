""" Functions for reading ascidoc files (*.adoc, *.asc, *.asciidoc) and tagging each line """
import os
import sys
import glob
import re
import logging
from shutil import copyfile

from pugnlp import futil
from nlpia.regexes import CRE_ACRONYM
# from nlpia.data_utils import iter_lines  # FIXME: reuse
from nlpia.constants import BOOK_PATH
from nlpia.regexes import RE_URL_SIMPLE, splitext
from nlpia.loaders import get_url_title, get_url_filemeta
from nlpia.transcoders import delimit_slug
from nlpia.translators import HyperlinkStyleCorrector
from nlpia.futil import rm_rf, rm_r  # noqa  (used in doctests to clean up)

logger = logging.getLogger(__name__)


# FIXME: redundant definitions here from develop branch
BLOCK_DELIMITERS = dict([('--', 'natural'), ('==', 'natural'), ('__', 'natural'), ('**', 'natural'),
                         ('++', 'latex'), ('////', 'comment')])
BLOCK_DELIM_CHRS = dict([(k[0], v) for k, v in BLOCK_DELIMITERS.items()])
BLOCK_DELIM_REGEXES = dict([(r'^[' + s[0] + r']{' + str(len(s)) + r',160}$', tag) for (s, tag) in BLOCK_DELIMITERS.items()])
BLOCK_HEADERS = dict([('[tip]', 'natural'), ('[note]', 'natural'), ('[important]', 'natural'), ('[quote]', 'natural')])
CRE_BLOCK_DELIMITER = re.compile('|'.join([s for s, tag in BLOCK_DELIM_REGEXES.items()]))
HEADER_TYPES = [('source', 'code'), ('latex', 'latex')]


# Working definitions from master branch
BLOCK_DELIMITERS = dict([('--', 'code'), ('==', 'natural_sidenote'), ('__', 'natural_quote'), ('**', 'natural_asside'),
                         ('++', 'latexmath'), ('//', 'comment')])
BLOCK_DELIMITER_CHRS = ''.join([k[0] for k in BLOCK_DELIMITERS.keys()])
BLOCK_HEADERS = dict([('[tip]', 'natural_tip'), ('[note]', 'natural_note'), 
                      ('[important]', 'natural_important'), ('[quote]', 'natural_quote')])
BLOCK_HEADERS4 = dict([(k[:4], v) for k, v in BLOCK_HEADERS.items()])

CRE_BLOCK_DELIMITER = re.compile(r'^[' + BLOCK_DELIMITER_CHRS + r']{2,50}$')
CRE_ANNOTATION = re.compile(r'^<([0-9]{1,2})>.*')
HEADER_TYPES = [('source', 'code'), ('latex', 'latex'), ('latexmath', 'latex'),
                ('template="glossary"', 'natural_glossary'), ("template='glossary'", 'natural_glossary')]

VALID_TAGS = set(['anchor', 'attribute', 'blank_line', 'block_header', 'caption', 'code', 'code_end', 'code_start', ] + 
                 [b for b in BLOCK_DELIMITERS.values()] + 
                 [b + '_start' for b in BLOCK_DELIMITERS.values()] + 
                 [b + '_end' for b in BLOCK_DELIMITERS.values()] + 
                 ['natural_heading{}'.format(i) for i in range(1, 6)] + 
                 ['image_link', 'natural', 'natural_end', 'natural_start', 'code_header'])
INCLUDE_TAGS = set(['natural', 'caption'] + ['natural_heading{}'.format(i) for i in range(1, 6)])
re_bad_footnotes = re.compile(r'footnote:\[' + RE_URL_SIMPLE + r'\]')


def get_lines(file_path=BOOK_PATH):
    r""" Retrieve text lines from the manuscript Chapter*.asc and Appendix*.asc files

    Args:
        file_path (str): Path to directory containing manuscript asciidoc files
        i.e.: /Users/cole-home/repos/nlpinaction/manuscript/ or nlpia.constants.BOOK_PATH

    Returns:
        list of lists of str, one list for each Chapter or Appendix

    >>> lines = get_lines(BOOK_PATH)
    >>> next(lines)
    ('.../src/nlpia/data/book/Appendix F -- Glossary.asc',
     ['= Glossary\n',
      '\n',
      "We've collected some ...])
    """
    if os.path.isdir(file_path):
        file_path = os.path.join(file_path, '*.asc')
        files = glob.glob(file_path)
    elif os.path.isfile(file_path):
        files = [file_path]
    elif '*' in file_path:
        if os.path.sep not in file_path:
            file_path = os.path.join(os.path.abspath(os.path.curdir), file_path)
        files = glob.glob(file_path)
    else:
        raise FileNotFoundError("Unable to find the directory or files requested.")
    lines = []
    for filepath in files:
        with open(filepath, 'r') as f:
            lines.append(f.readlines())
    return zip(files, lines)


def get_acronyms(manuscript=os.path.expanduser('~/code/nlpia/lane/manuscript')):
    """ Find all the 2 and 3-letter acronyms in the manuscript and return as a sorted list of tuples """
    acronyms = []
    for f, lines in get_lines(manuscript):
        for line in lines:
            matches = CRE_ACRONYM.finditer(line)
            if matches:
                for m in matches:
                    if m.group('a2'):
                        acronyms.append((m.group('a2'), m.group('s2')))
                    elif m.group('a3'):
                        acronyms.append((m.group('a3'), m.group('s3')))
                    elif m.group('a4'):
                        acronyms.append((m.group('a4'), m.group('s4')))
                    elif m.group('a5'):
                        acronyms.append((m.group('a5'), m.group('s5')))

    return sorted(dict(acronyms).items())


def write_glossary(manuscript=os.path.expanduser('~/code/nlpia/lane/manuscript'), linesep=None):
    """ Compose an asciidoc string with acronyms culled from the manuscript """
    linesep = linesep or os.linesep
    lines = ['[acronyms]', '== Acronyms', '', '[acronyms,template="glossary",id="terms"]']
    acronyms = get_acronyms(manuscript)
    for a in acronyms:
        lines.append('*{}*:: {} -- '.format(a[0], a[1][0].upper() + a[1][1:]))
    return linesep.join(lines)


def tag_lines(lines, include_tags=None):
    r""" Naively tags lines from manuscript with: code, natural, heading, etc.

    Returns:
        list of tuples  [(tag, line), ...]

    >>> ' '.join(sorted(VALID_TAGS))
    'anchor attribute blank_line block_header caption code code_end code_header code_start
     comment comment_end comment_start image_link latexmath latexmath_end latexmath_start
     natural natural_asside natural_asside_end natural_asside_start natural_end
     natural_heading1 natural_heading2 natural_heading3 natural_heading4 natural_heading5
     natural_quote natural_quote_end natural_quote_start
     natural_sidenote natural_sidenote_end natural_sidenote_start natural_start'
    >>> list(tag_lines('|= Title| :chapter: 0|Hello|cruel world|==Heading Level 2| \t| [source,bash]|====|$ grep this|====|'\
    ...                .split('|')))
    [('blank_line', ''), ('natural_heading1', '= Title'), ('attribute', ' :chapter: 0'),
     ('natural', 'Hello'), ('natural', 'cruel world'), ('natural_heading2', '==Heading Level 2'),
     ('blank_line', ' \t'), ('code_header', ' [source,bash]'),
     ('code_start', '===='), ('code', '$ grep this'), ('code_end', '===='), ('blank_line', '')]
    """
    current_block_type = None
    block_terminator = None
    tag = ''
    tagged_lines = []
    for idx, line in enumerate(lines):
        normalized_line = line.lower().strip().replace(" ", "")
        # [source,...] with or without any following "----" block delimiter
        # TODO: make this a regex that classifies among the different types (source, glossary, tip, etc)
        header_type = next((HEADER_TYPES[i] for i in range(len(HEADER_TYPES)) if
                            normalized_line.startswith('[') and normalized_line[1:].startswith(HEADER_TYPES[i][0])),
                           None)
        if header_type:
            current_block_type = header_type[1]
            tag = current_block_type + '_header'
            block_terminator = None
        elif normalized_line[:4] in BLOCK_HEADERS4:
            current_block_type = BLOCK_HEADERS4[normalized_line[:4]]
            tag = current_block_type + '_header'  # BLOCK_HEADERS[normalized_line]
            block_terminator = None
        elif (
                CRE_BLOCK_DELIMITER.match(normalized_line) and
                normalized_line[:2] in BLOCK_DELIMITERS):  # or (tag in set('caption anchor'.split()))):
            if (not idx or not current_block_type or not block_terminator):
                current_block_type = (current_block_type or BLOCK_DELIMITERS[normalized_line[:2]])
                tag = current_block_type + '_start'
                block_terminator = normalized_line
            elif block_terminator and line.rstrip() == block_terminator:
                tag = current_block_type + '_end'
                current_block_type = None
                block_terminator = None
            else:
                tag = current_block_type
        elif current_block_type and (line.rstrip() == block_terminator or 
                                     (not block_terminator and not normalized_line)):
            tag = current_block_type + '_end'
            current_block_type = None
            block_terminator = None
        elif current_block_type:
            tag = current_block_type
        elif not normalized_line:
            tag = 'blank_line'
        elif normalized_line.startswith(r'//'):
            tag = 'comment'
        elif normalized_line.startswith(r':'):
            tag = 'attribute'
        elif normalized_line.startswith('='):
            tag = 'natural_heading'
            tag += str(len([c for c in normalized_line[:6].split()[0] if c == '=']))
        elif normalized_line.startswith('.'):
            tag = 'caption'
        elif normalized_line.startswith('image:'):
            tag = 'image_link'
        elif normalized_line.startswith('[['):
            tag = 'anchor'
        else:
            tag = 'natural'
            current_block_type = None

        tagged_lines.append((tag, line))

    return filter_tagged_lines(tagged_lines, include_tags=include_tags)


def get_tagged_sections(book_dir=BOOK_PATH, include_tags=None):
    """ Get list of (adoc_file_path, (adoc_syntax_tag, raw_line_str))

    >>> get_tagged_sections()
    [('...src/nlpia/data/book/Appendix F -- Glossary.asc', <generator object filter_tagged_lines at ...>)]
    """
    return [(filepath, tag_lines(lines, include_tags=include_tags)) for filepath, lines in get_lines(book_dir)]


def find_bad_footnote_urls(tagged_lines, include_tags=None):
    """ Find lines in the list of 2-tuples of adoc-tagged lines that contain bad footnotes (only urls) 

    >>> sections = get_tagged_sections(BOOK_PATH)
    >>> tagged_lines = list(sections[0][1])
    >>> find_bad_footnote_urls(tagged_lines)
    [[30, 'https://spacy.io/usage/linguistic-features#rule-based-morphology']]
    """
    section_baddies = []
    logger.debug(tagged_lines[:2])
    for lineno, (tag, line) in enumerate(tagged_lines):
        line_baddies = None
        if tag is None or include_tags is None or tag in include_tags or any((tag.startswith(t) for t in include_tags)):
            line_baddies = get_line_bad_footnotes(line=line, tag=tag)
        if line_baddies and len(line_baddies) > 1:
            section_baddies.append([lineno] + line_baddies[1:])
        else:
            pass
            # section_baddies.append(line)
    return section_baddies


# def find_all_bad_footnote_urls(book_dir=BOOK_PATH, include_tags=['natural']):
#     """ Find lines in the manuscript that contain bad footnotes (only urls) """
#     sections = get_tagged_sections(book_dir=book_dir, include_tags=include_tags)
#     bad_url_lines = {}
#     for fileid, (filepath, tagged_lines) in enumerate(sections):
#         section_baddies = find_bad_footnote_urls(tagged_lines, include_tags=include_tags)
#         if section_baddies:
#             bad_url_lines[filepath] = section_baddies
#     return bad_url_lines


def infer_url_title(url):
    """ Guess what the page title is going to be from the path and FQDN in the URL

    >>> infer_url_title('https://ai.googleblog.com/2018/09/the-what-if-tool-code-free-probing-of.html')
    'the what if tool code free probing of'
    """
    meta = get_url_filemeta(url)
    title = ''
    if meta:
        if meta.get('hostname', url) == 'drive.google.com':
            title = get_url_title(url)
        else:
            title = meta.get('filename', meta['hostname']) or meta['hostname']
            title, fileext = splitext(title)
    else:
        logging.error('Unable to retrieve URL: {}'.format(url))
        return None
    return delimit_slug(title, ' ') 


def get_line_bad_footnotes(line, tag=None, include_tags=None):
    """ Return [original_line, url_footnote1, url_footnote2, ... url_footnoteN] for N bad footnotes in the line """ 
    if tag is None or include_tags is None or tag in include_tags or any((tag.startswith(t) for t in include_tags)):
        found_baddies = re_bad_footnotes.findall(line)
        return [line] + [baddie[0] for baddie in found_baddies]
    return [line]


def translate_line_footnotes(line, tag=None, default_title='<NOT_FOUND>'):
    r""" Find all bare-url footnotes, like "footnote:[moz.org]" and add a title like "footnote:[Moz (moz.org)]" 

    >>> translate_line_footnotes('*Morphemes*:: Parts of tokens or words that contain meaning in and of themselves.'\
    ...     'footnote:[https://spacy.io/usage/linguistic-features#rule-based-morphology]')
    '*Morphemes*:: Parts of tokens or words that contain meaning in and of
     themselves.footnote:[See the web page titled "Linguistic Features : spaCy Usage Documentation"
     (https://spacy.io/usage/linguistic-features#rule-based-morphology).]'
    """
    line_urls = get_line_bad_footnotes(line, tag=tag)
    urls = line_urls[1:] if line_urls else []
    for url in urls:
        footnote = 'footnote:[{url}]'.format(url=url)
        new_footnote = footnote
        # TODO: use these to extract name from hyperlinks
        title = get_url_title(url)
        title = title or infer_url_title(url)
        title = (title or '').strip(' \t\n\r\f-_:|="\'/\\')
        title = title if ' ' in (title or 'X') else None

        if title:
            brief_title = title.split('\n')[0].strip().split('|')[0].strip().split('Â')[0].strip().split('·')[0].strip()
            logging.info('URL: {}'.format(url))
            logging.info('TITLE: {}'.format(title))
            title = brief_title if len(brief_title) > 3 and len(title) > 55 else title
            title = title.replace('Â', '').replace('·', ':').replace('|', ':').replace('\n', '--')
            logging.info('FINAL: {}'.format(title))
        title = title or default_title
        if title:
            new_footnote = 'footnote:[See the web page titled "{title}" ({url}).]'.format(title=(title or default_title), url=url)
        elif title is None:
            logging.error('Unable to find a title for url: {}'.format(url))
        else:
            new_footnote = 'footnote:[See the web page ({url}).]'.format(url=url)
        line = line.replace(
            footnote,
            new_footnote)

    return line


def ensure_dir_exists(dest):
    if dest is not None:
        dest = dest.rstrip(os.path.sep)
        if os.path.isdir(dest):
            pass
        elif os.path.isdir(os.path.dirname(dest)):
            parent = os.path.dirname(dest)
            dest = os.path.join(parent, dest[len(parent):].lstrip(os.path.sep))
        elif os.path.sep in dest:
            raise FileNotFoundError('Unable to find destination directory for the path: {}'.format(dest))
        elif splitext(dest)[1]:
            raise FileNotFoundError(
                'Unable to find destination directory for the path. It looks like a file path rather than a directory: {}'.format(
                    dest))
        if not os.path.isdir(dest):
            logger.warning('Creating directory with mkdir_p({})'.format(repr(dest)))
        futil.mkdir_p(dest)
        logger.info('Saving translated files in {}{}*'.format(dest, os.path.sep))
    return dest


def translate_book(translators=(HyperlinkStyleCorrector().translate, translate_line_footnotes),
                   book_dir=BOOK_PATH, dest=None, include_tags=None,
                   ext='.nlpiabak', skip_untitled=True):
    """ Fix any style corrections listed in `translate` list of translation functions

    >>> len(translate_book(book_dir=BOOK_PATH, dest='cleaned_hyperlinks'))
    3
    >>> rm_rf(os.path.join(BOOK_PATH, 'cleaned_hyperlinks'))
    """
    if callable(translators) or not hasattr(translators, '__len__'):
        translators = (translators,)

    sections = get_tagged_sections(book_dir=book_dir, include_tags=include_tags)
    file_line_maps = []

    for fileid, (filepath, tagged_lines) in enumerate(sections):
        logger.info('filepath={}'.format(filepath))
        destpath = filepath
        if not dest:
            copyfile(filepath, filepath + '.' + ext.lstrip('.'))
        elif os.path.sep in dest:
            destpath = os.path.join(dest, os.path.basename(filepath))
        else:
            destpath = os.path.join(os.path.dirname(filepath), dest, os.path.basename(filepath))
        ensure_dir_exists(os.path.dirname(destpath))
        with open(destpath, 'w') as fout:
            logger.info('destpath={}'.format(destpath))
            for lineno, (tag, line) in enumerate(tagged_lines):
                if (include_tags is None or tag in include_tags or
                        any((tag.startswith(t) for t in include_tags))):
                    for translate in translators:
                        new_line = translate(line)  # TODO: be smarter about writing to files in-place
                        if line != new_line:
                            file_line_maps.append((fileid, lineno, filepath, destpath, line, new_line))
                            line = new_line
                fout.write(line)
    return file_line_maps


def correct_hyperlinks(book_dir=BOOK_PATH, dest=None, include_tags=None,
                       ext='.nlpiabak', skip_untitled=True):
    """ DEPRECATED (see translate_line_footnotes)

    Find bad footnotes (only urls), visit the page, add the title to the footnote 

    >>> len(correct_hyperlinks(book_dir=BOOK_PATH, dest='cleaned_hyperlinks'))
    2
    >>> rm_rf(os.path.join(BOOK_PATH, 'cleaned_hyperlinks'))
    """
    # bad_url_lines = find_all_bad_footnote_urls(book_dir=book_dir)
    # file_line_maps = []
    return translate_book(translators=HyperlinkStyleCorrector().translate,
                          book_dir=book_dir, dest=dest, include_tags=include_tags,
                          ext=ext, skip_untitled=skip_untitled)


def correct_bad_footnote_urls(book_dir=BOOK_PATH, dest=None, include_tags=None,
                              ext='.nlpiabak', skip_untitled=True):
    """ DEPRECATED (see translate_line_footnotes)

    Find bad footnotes (only urls), visit the page, add the title to the footnote 

    >>> len(correct_bad_footnote_urls(book_dir=BOOK_PATH, dest='cleaned_footnotes'))
    1
    >>> rm_r(os.path.join(BOOK_PATH, 'cleaned_footnotes'))
    """
    # bad_url_lines = find_all_bad_footnote_urls(book_dir=book_dir)
    # file_line_maps = []
    return translate_book(translators=translate_line_footnotes, book_dir=book_dir, dest=dest, include_tags=include_tags,
                          ext=ext, skip_untitled=skip_untitled)


def filter_lines(input_file, output_file, translate=lambda line: line):
    """ Translate all the lines of a single file """
    filepath, lines = get_lines([input_file])[0]
    return filepath, [(tag, translate(line=line, tag=tag)) for (tag, line) in lines]


def filter_tagged_lines(tagged_lines, include_tags=None, exclude_tags=None):
    r""" Return iterable of tagged lines where the tags all start with one of the include_tags prefixes

    >>> filter_tagged_lines([('natural', "Hello."), ('code', '[source,python]'), ('code', '>>> hello()')])
    <generator object filter_tagged_lines at ...>
    >>> list(filter_tagged_lines([('natural', "Hello."), ('code', '[source,python]'), ('code', '>>> hello()')],
    ...      include_tags='natural'))
    [('natural', 'Hello.')]
    """
    include_tags = (include_tags,) if isinstance(include_tags, str) else include_tags
    exclude_tags = (exclude_tags,) if isinstance(exclude_tags, str) else exclude_tags
    for tagged_line in tagged_lines:
        if (include_tags is None or tagged_line[0] in include_tags or
                any((tagged_line[0].startswith(t) for t in include_tags))):
            if exclude_tags is None or not any((tagged_line[0].startswith(t) for t in exclude_tags)):
                yield tagged_line
            else:
                logger.debug('skipping tag {} because it starts with one of the exclude_tags={}'.format(
                    tagged_line[0], exclude_tags))

        else:
            logger.debug('skipping tag {} because not in {}'.format(tagged_line[0], include_tags))


def main(book_dir=BOOK_PATH, include_tags=None, verbosity=1):
    r""" Parse all the asciidoc files in book_dir, returning a list of 2-tuples of lists of 2-tuples (tagged lines) 

    >>> main(BOOK_PATH, verbosity=0)
    [('.../src/nlpia/data/book/Appendix F -- Glossary.asc', <generator object filter_tagged_lines at ...>)]
    >>> main(BOOK_PATH, include_tags='natural', verbosity=1)
    = Glossary
    We've collected some definitions of some common NLP and ML acronyms and terminology here.footnote:[Bill Wilson...
    at the university of New South Wales in Australia has a more complete one here:...
    https://www.cse.unsw.edu.au/~billw/nlpdict.html]...
    You can find some of the tools we used to generate this list in the `nlpia` python package at...
    ...
    >>> tagged_lines = list(main(BOOK_PATH, include_tags=['natural', 'blank'], verbosity=0))
    >>> len(tagged_lines[0])
    2
    >>> tagged_lines = list(main(BOOK_PATH, include_tags=['natural', 'blank'], verbosity=1))
    = Glossary
    <BLANKLINE>
    We've collected some definitions of some common NLP and ML acronyms and terminology here.footnote:[...
    >>> tagged_lines = list(main(BOOK_PATH, include_tags='natural', verbosity=1))
    = Glossary
    We've collected some definitions of some common NLP and ML acronyms and terminology here.footnote:[...

    TODO:
       `def filter_tagged_lines(tagged_lines)` that returns an iterable. 
    """
    if verbosity:
        logger.info('book_dir: {}'.format(book_dir))
        logger.info('include_tags: {}'.format(include_tags))
        logger.info('verbosity: {}'.format(verbosity))

    include_tags = [include_tags] if isinstance(include_tags, str) else include_tags
    include_tags = None if not include_tags else set([t.lower().strip() for t in include_tags])
    sections = get_tagged_sections(book_dir=book_dir)
    if verbosity >= 1:
        for filepath, tagged_lines in sections:
            tagged_lines = filter_tagged_lines(tagged_lines, include_tags=include_tags)
            if verbosity > 1:
                print('=' * 75)
                print(filepath)
                print('-' * 75)
            if verbosity == 1:
                for tag, line in tagged_lines:
                    print(line)
            else:
                for tagged_line in tagged_lines:
                    print(tagged_line)
            if verbosity > 1:
                print('=' * 79)
                print()
    else:
        logger.debug('vebosity={} so nothing output to stdout with print()'.format(verbosity))
    return sections


if __name__ == '__main__':
    args = sys.argv[1:]

    book_dir = os.path.curdir
    verbosity = 1
    include_tags = tuple(INCLUDE_TAGS)

    if args:
        book_dir = args[0]
        args = args[1:]

    if args:
        try:
            verbosity = int(args[0])
            include_tags = args[1:] or include_tags
        except ValueError:
            verbosity = 1
            include_tags = args

    if include_tags and include_tags[0].strip().lower()[: 3] in ('all', 'none', 'true'):
        include_tags = None

    # print('Parsing Chapters and Appendices in: ' + book_dir)
    # print('***PRINTING LINES WITH TAGS***: ' + str(include_tags))
    main(book_dir=book_dir, include_tags=include_tags, verbosity=verbosity)
