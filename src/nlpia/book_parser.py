import os
import sys
import glob


def get_lines(file_path):
    '''
    args:
        file_path  string  aboslute path of the Manning manuscript asciidoc files
                           i.e.: /Users/cole-home/repos/nlpinaction/manuscript

    Returns:
        list of lists of str, one list for each Chapter or Appendix
    '''
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

    Possible Tags:
        blank_line
        attribute
        source_header
        block_header
        block_start
        block_end
        code
        natural
        heading1 ... heading6
        image_link
        caption
        code

    Returns:
        list of tuples  [(tag, line), ...]

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
        elif normalized_line[:4] in ('[tip', '[not', '[imp'):
            current_block_type = 'text'
            block_start = idx
            open_block = True
            tag = 'block_header'
        elif open_block and idx == block_start + 1:
            if not normalized_line.startswith('--') and not normalized_line.startswith('=='):
                block_terminator = '\n'
            else:
                block_terminator = normalized_line[:2]
            tag = (current_block_type or 'block') + '_start'
        elif open_block and normalized_line[:2] == block_terminator:
            current_block_type = None
            open_block = False
            block_terminator = None
            block_start = 0
            tag = 'block_end'
        elif open_block and current_block_type == 'code':
            tag = 'code'
        elif normalized_line.startswith('='):
            tag = 'heading'
            tag += str(len([c for c in normalized_line if c == '=']))
        elif normalized_line.startswith('.'):
            tag = 'caption'
        elif normalized_line.startswith('image:'):
            tag = 'image_link'
        else:
            tag = 'natural'
            current_block_type = None

        tup_lines.append((tag, line.strip()))

    return tup_lines


if __name__ == '__main__':
    args = sys.argv[1:]
    book_dir = os.path.curdir
    if args:
        book_dir = args[0]
    include_tags = ['natural']
    if len(args) > 1:
        include_tags = args[1:]

    for section in get_lines(book_dir):
        for line in tag_lines(section):
            if line[0] in include_tags:
                print(line[1])
