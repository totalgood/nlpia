import os
import sys
import glob
import re
import logging

from nlpia.constants import BOOK_PATH
from nlpia.regexes import RE_URL_SIMPLE
from nlpia.loaders import get_url_title


logger = logging.getLogger(__name__)
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


def get_lines(file_path=BOOK_PATH):
    r""" Retrieve text lines from the manuscript Chapter*.asc and Appendix*.asc files

    Args:
        file_path (str): Path to directory containing manuscript asciidoc files
        i.e.: /Users/cole-home/repos/nlpinaction/manuscript/ or nlpia.constants.BOOK_PATH

    Returns:
        list of lists of str, one list for each Chapter or Appendix

    >>> lines = get_lines(os.path.join(BOOK_PATH))
    >>> next(lines)
    ('.../src/nlpia/data/book/Appendix F -- Glossary.asc',
     ['= Glossary\n',
      '\n',
      "We've collected some ...
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
    lines = []
    for file in files:
        with open(file, 'r') as f:
            lines.append(f.readlines())
    return zip(files, lines)


def tag_lines(lines):
    r""" Naively tags lines from manuscript with: code, natural, heading, etc.

    Returns:
        list of tuples  [(tag, line), ...]

    >> VALID_TAGS == {'anchor', 'attribute', 'blank_line', 'block_header', 'caption', 'code', 'code_end', 'code_start',
    ... 'comment', 'comment_end', 'comment_start',
    ... 'natural_heading1', 'natural_heading2', 'natural_heading3', 'natural_heading4', 'natural_heading5',
    ... 'image_link', 'natural', 'natural_end', 'natural_start', 'source_header'}
    True
    >>> tag_lines('|= Title| :chapter: 0|Hello|cruel world|==Heading Level 2| \t| [source,bash]|====|$ grep this|====|'.split('|'))
    [('blank_line', ''), ('natural_heading1', '= Title'), ('attribute', ':chapter: 0'), ('natural', 'Hello'),
     ('natural', 'cruel world'), ('natural_heading2', '==Heading Level 2'), ('blank_line', ''),
     ('code_header', '[source,bash]'), ('code_start', '===='), ('code', '$ grep this'), ('code_end', '===='),
     ('blank_line', '')]
    """
    current_block_type = None
    block_terminator = None
#    block_start = None
    tag = ''
    tagged_lines = []
    for idx, line in enumerate(lines):
        # print(current_block_type)
        # print(line)
        normalized_line = line.lower().strip().replace(" ", "")

        # [source,...] with or without any following "----" block delimiter
        # TODO: make this a regex that classifies among the different types (source, glossary, tip, etc)
        header_type = next((HEADER_TYPES[i] for i in range(len(HEADER_TYPES)) if
                            normalized_line.startswith('[') and normalized_line[1:].startswith(HEADER_TYPES[i][0])),
                           None)
        if header_type:
            current_block_type = header_type[1]
#            block_start = idx
            tag = current_block_type + '_header'
            block_terminator = None
        # [note],[quote],[important],... etc with or without any following "====" block delimiter
        elif normalized_line[:4] in BLOCK_HEADERS4:
            current_block_type = BLOCK_HEADERS4[normalized_line[:4]]
#            block_start = idx
            tag = current_block_type + '_header'  # BLOCK_HEADERS[normalized_line]
            block_terminator = None
        # # "==", "--", '__', '++' block delimiters below block type ([note], [source], or blank line)
        # elif current_block_type
        #     if idx == block_start + 1:
        #         if line.strip()[:2] in ('--', '==', '__', '**'):
        #             block_terminator = line.strip()
        #             tag = current_block_type + '_start'
        #         elif line.strip()[:2] == '++':
        #             block_terminator = line.strip()
        #             tag = current_block_type + '_start'
        #         # # this should only happen when there's a "bare" untyped block that has started
        #         # else:
        #         #     block_terminator = line.strip()
        #         #     tag = current_block_type
        #     elif:

        # block start and end delimiters: '----', '====', '____', '****', '--' etc
        # block delimiters (like '----') can start a block with or without a block type already defined
        elif (
                CRE_BLOCK_DELIMITER.match(normalized_line) and
                normalized_line[:2] in BLOCK_DELIMITERS):  # or (tag in set('caption anchor'.split()))):
            if (not idx or not current_block_type or not block_terminator):
                current_block_type = (current_block_type or BLOCK_DELIMITERS[normalized_line[:2]])
#                block_start = idx 
                tag = current_block_type + '_start'
                block_terminator = normalized_line
            elif block_terminator and line.rstrip() == block_terminator:
                tag = current_block_type + '_end'
                current_block_type = None
                block_terminator = None
#                block_start = None  # block header not allowed on line 0
            else:
                tag = current_block_type
        elif current_block_type and (line.rstrip() == block_terminator or 
                                     (not block_terminator and not normalized_line)):
            tag = current_block_type + '_end'
            current_block_type = None
            block_terminator = None
#            block_start = None  # block header not allowed on line 0
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

        tagged_lines.append((tag, line.strip()))

    return tagged_lines


def get_tagged_sections(book_dir):
    return [(filepath, tag_lines(lines)) for filepath, lines in get_lines(book_dir)]


def find_bad_footnote_urls(book_dir=os.path.curdir, include_tags=['natural']):
    """ Find lines in the manuscript that contain bad footnotes (only urls) """
    re_bad_footnotes = re.compile(r'footnote:\[' + RE_URL_SIMPLE + r'\]')
    sections = get_tagged_sections(book_dir=book_dir)
    bad_url_lines = []
    for filepath, tagged_lines in sections:
        for tag, line in tagged_lines:
            if include_tags is None or tag in include_tags or \
                    any((tag.startswith(t) for t in include_tags)):
                found_baddies = re_bad_footnotes.findall(line)
                if found_baddies:
                    bad_url_lines.append([line] + [baddie[0] for baddie in found_baddies])
    return bad_url_lines


def correct_bad_footnote_urls(book_dir=os.path.curdir, include_tags=['natural'], skip_untitled=True):
    """ Find bad footnotes (only urls), visit the page, add the title to the footnote 

    >>> correct_bad_footnote_urls(BOOK_PATH)
    [['*Morphemes*:: Parts of tokens or words that contain meaning in and of themselves. The morphemes ...
      ('https://spacy.io/usage/linguistic-features#rule-based-morphology',
       'Linguistic Features Â· spaCy Usage Documentation')]]
    """
    bad_url_lines = find_bad_footnote_urls(book_dir=book_dir)
    url_line_titles = []
    for line in bad_url_lines:
        line_titles = [line[0]]
        for url in line[1:]:
            title = get_url_title(url)
            if not skip_untitled or title: 
                line_titles.append((url, title))
        if len(line_titles) > 1:
            url_line_titles.append(line_titles)

    return url_line_titles


def main(book_dir=os.path.curdir, include_tags=None, verbosity=1):
    r""" Parse all the asciidoc files in book_dir, returning a list of 2-tuples of lists of 2-tuples (tagged lines) 

    >>> main(BOOK_PATH, verbosity=0)
    [('...src/nlpia/data/book/Appendix F -- Glossary.asc',
      [('natural_heading1', '= Glossary'),
       ('blank_line', ''),
       ('natural',
        "We've collected some definitions ...
    >>> main(BOOK_PATH, include_tags='natural', verbosity=1)
    = Glossary
    We've collected some definitions of some common NLP and ML acronyms and terminology here.footnote:[Bill Wilson...
    at the university of New South Wales in Australia has a more complete one here:...
    https://www.cse.unsw.edu.au/~billw/nlpdict.html]...
    You can find some of the tools we used to generate this list in the `nlpia` python package at...
    ...
    >>> tagged_lines = main(BOOK_PATH, include_tags=['natural', 'blank'], verbosity=0)
    >>> tagged_lines = main(BOOK_PATH, include_tags=['natural', 'blank'], verbosity=1)
    = Glossary
    <BLANKLINE>
    We've collected some definitions of some common NLP and ML acronyms and terminology here.footnote:[...
    >>> tagged_lines = main(BOOK_PATH, include_tags='natural', verbosity=1)
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
    if verbosity > 0:
        for filepath, tagged_lines in sections:
            if verbosity > 1:
                print('=' * 75)
                print(filepath)
                print('-' * 75)
            for tagged_line in tagged_lines:
                if include_tags is None or tagged_line[0] in include_tags or \
                        any((tagged_line[0].startswith(t) for t in include_tags)):
                    if verbosity == 1:
                        print(tagged_line[1])
                    if verbosity > 1:
                        print(tagged_line)
                else:
                    logger.debug('skipping tag {} because not in {}'.format(tagged_line[0], include_tags))
            if verbosity > 1:
                print('=' * 79)
                print()
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
