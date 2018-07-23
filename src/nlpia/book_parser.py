import os
import sys
import glob


VALID_TAGS = tuple('natural caption comment blank_line attribute source_header block_header code anchor image_link'.split() +
                   'code_start code_end natural_start natural_end'.split() +
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
    block_terminator = None
    block_start = 0
    tagged_lines = []
    for idx, line in enumerate(lines):
        normalized_line = line.lower().strip().replace(" ", "")

        if normalized_line.startswith('[source'):
            current_block_type = 'code'
            block_start = idx
            tag = 'source_header'
        elif normalized_line[:4] in ('[tip', '[not', '[imp', '[quo'):
            current_block_type = 'natural'
            block_start = idx
            tag = 'block_header'
        elif current_block_type and idx == block_start + 1:
            if normalized_line.startswith('--') or normalized_line.startswith('=='):
                block_terminator = normalized_line[:2]
                tag = current_block_type + '_start'
            else:
                block_terminator = ''
                tag = current_block_type
        elif current_block_type and line.lstrip()[:2] == block_terminator:
            tag = current_block_type + '_end'
            current_block_type = None
            block_terminator = None
            block_start = 0
        elif current_block_type:
            tag = current_block_type
        if not normalized_line:
            tag = 'blank_line'
        elif normalized_line.startswith(r'//'):
            tag = 'comment'
        elif normalized_line.startswith(r':'):
            tag = 'attribute'
        elif normalized_line.startswith('='):
            tag = 'heading'
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


def main(book_dir='.', include_tags=INCLUDE_TAGS, verbosity=1):
    sections = [tag_lines(section) for section in get_lines(book_dir)]
    if verbosity > 0:
        for section in sections:
            for tag_line in section:
                if tag_line[0] in include_tags:
                    if verbosity == 1:
                        print(tag_line[1])
                    if verbosity > 1:
                        print(tag_line)
    return sections


if __name__ == '__main__':
    args = sys.argv[1:]
    book_dir = os.path.curdir
    if args:
        book_dir = args[0]
    include_tags = INCLUDE_TAGS
    if len(args) > 1:
        include_tags = list(args[1:])
    # print('Parsing Chapters and Appendices in: ' + book_dir)
    # print('***PRINTING LINES WITH TAGS***: ' + str(include_tags))
    main(book_dir=book_dir, include_tags=include_tags, verbosity=1)
