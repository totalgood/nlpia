import os
import sys
import glob


VALID_TAGS = tuple('natural caption blank_line attribute source_header block_header code anchor image_link'.split() +
                   'block_start block_end code_start code_end natural_start natural_end'.split() +
                   ['heading{}'.format(i) for i in range(1, 7)])
INCLUDE_TAGS = ('natural', 'caption', 'heading1', 'heading2', 'heading3', 'heading4', 'heading5')


def get_lines(file_path):
    r""" Retrieve text lines from the manuscript Chapter*.asc and Appendix*.asc files

    Args:
        file_path (str): Path to directory containing manuscript asciidoc files
        i.e.: /Users/cole-home/repos/nlpinaction/manuscript/

    Returns:
        list of lists of str, one list for each Chapter or Appendix
    """
    path = os.path.join(file_path, 'Chapter*')
    files = glob.glob(path)
    lines = []
    for file in files:
        with open(file, 'r') as f:
            lines.append(f.readlines())

    path = os.path.join(file_path, 'Appendix*')
    files = glob.glob(path)
    for file in files:
        with open(file, 'r') as f:
            lines.append(f.readlines())
    return lines


def tag_lines(lines):
    r""" Naively tags lines from manuscript with: code, natural, heading, etc.

    Returns:
        list of tuples  [(tag, line), ...]

    >>> VALID_TAGS
    ('natural',
     'caption',
     'blank_line',
     'attribute',
     'source_header',
     'block_header',
     'code',
     'anchor',
     'image_link',
     'block_start',
     'block_end',
     'code_start',
     'code_end',
     'natural_start',
     'natural_end',
     'heading1',
     'heading2',
     'heading3',
     'heading4',
     'heading5',
     'heading6')

    >>> tag_lines('|= Title| :chapter: 0|Hello|cruel world|==Heading Level 2| \t| [source,bash]|====|$ grep this|====|'.split('|'))
    [('blank_line', ''),
     ('heading1', '= Title'),
     ('attribute', ':chapter: 0'),
     ('natural', 'Hello'),
     ('natural', 'cruel world'),
     ('heading2', '==Heading Level 2'),
     ('blank_line', ''),
     ('source_header', '[source,bash]'),
     ('block_start', '===='),
     ('code', '$ grep this'),
     ('block_end', '===='),
     ('blank_line', '')]
    """
    current_block_type = None
    open_block = False
    block_terminator = None
    block_start = 0
    tup_lines = []
    for idx, line in enumerate(lines):
        normalized_line = line.lower().strip().replace(" ", "")

        if not normalized_line:
            tag = 'blank_line'
        elif normalized_line[0] in r'/:':
            tag = 'attribute'
        elif normalized_line.startswith('[source'):
            current_block_type = 'code'
            block_start = idx
            open_block = True
            tag = 'source_header'
        elif normalized_line[:4] in ('[tip', '[not', '[imp', '[quo'):
            current_block_type = 'natural'
            block_start = idx
            open_block = True
            tag = 'block_header'
        elif open_block and idx == block_start + 1:
            if not normalized_line.startswith('--') and not normalized_line.startswith('=='):
                block_terminator = '\n'
                tag = current_block_type
            else:
                block_terminator = normalized_line[:2]
                tag = (current_block_type or 'block') + '_start'
        elif open_block and normalized_line[:2] == block_terminator:
            current_block_type = None
            open_block = False
            block_terminator = None
            block_start = 0
            tag = (current_block_type or 'block') + '_end'
        elif open_block and current_block_type == 'code':
            tag = 'code'
        elif normalized_line.startswith('='):
            tag = 'heading'
            tag += str(len([c for c in normalized_line if c == '=']))
        elif normalized_line.startswith('.'):
            tag = 'caption'
        elif normalized_line.startswith('image:'):
            tag = 'image_link'
        elif normalized_line.startswith('[['):
            tag = 'anchor'
        else:
            tag = 'natural'
            current_block_type = None

        tup_lines.append((tag, line.strip()))

    return tup_lines


def main(book_dir='.',
         include_tags=INCLUDE_TAGS,
         verbose=True):
    sections = [tag_lines(section) for section in get_lines(book_dir)]
    if verbose:
        for section in sections:
            for line in section:
                if line[0] in include_tags:
                    print(line[1])
    return sections


if __name__ == '__main__':
    args = sys.argv[1:]
    book_dir = os.path.curdir
    if args:
        book_dir = args[0]
    include_tags = ['natural']
    if len(args) > 1:
        include_tags = list(args[1:])
    # print('Parsing Chapters and Appendices in: ' + book_dir)
    # print('***PRINTING LINES WITH TAGS***: ' + str(include_tags))
    main(book_dir=book_dir, include_tags=include_tags, verbose=True)
