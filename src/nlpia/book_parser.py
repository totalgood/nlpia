import os
import sys
import glob


VALID_TAGS = {'anchor', 'attribute', 'blank_line', 'block_header', 'caption', 'code', 'code_end', 'code_start',
              'comment', 'comment_end', 'comment_start',
              'heading1', 'heading2', 'heading3', 'heading4', 'heading5',
              'image_link', 'natural', 'natural_end', 'natural_start', 'source_header'}
INCLUDE_TAGS = {'natural', 'caption', 'heading1', 'heading2', 'heading3', 'heading4', 'heading5'}


def get_lines(file_path):
    r""" Retrieve text lines from the manuscript Chapter*.asc and Appendix*.asc files

    Args:
        file_path (str): Path to directory containing manuscript asciidoc files
        i.e.: /Users/cole-home/repos/nlpinaction/manuscript/

    Returns:
        list of lists of str, one list for each Chapter or Appendix
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


BLOCK_DELIMITERS = dict([('----', 'natural'), ('====', 'natural'), ('____', 'natural'), ('****', 'natural'),
                         ('++++', 'latex'), ('////', 'comment')])
BLOCK_HEADERS = dict([('[tip]', 'natural'), ('[note]', 'natural'), ('[important]', 'natural'), ('[quote]', 'natural')])


def tag_lines(lines):
    r""" Naively tags lines from manuscript with: code, natural, heading, etc.

    Returns:
        list of tuples  [(tag, line), ...]

    >>> VALID_TAGS == {'anchor', 'attribute', 'blank_line', 'block_header', 'caption', 'code', 'code_end', 'code_start',
    ... 'comment', 'comment_end', 'comment_start',
    ... 'heading1', 'heading2', 'heading3', 'heading4', 'heading5',
    ... 'image_link', 'natural', 'natural_end', 'natural_start', 'source_header'}
    True
    >>> tag_lines('|= Title| :chapter: 0|Hello|cruel world|==Heading Level 2| \t| [source,bash]|====|$ grep this|====|'.split('|'))
    [('blank_line', ''), ('heading1', '= Title'), ('attribute', ':chapter: 0'), ('natural', 'Hello'),
    ('natural', 'cruel world'), ('heading2', '==Heading Level 2'), ('blank_line', ''),
    ('source_header', '[source,bash]'),
    ('code_start', '===='), ('code', '$ grep this'), ('code_end', '===='), ('blank_line', '')]
    """
    current_block_type = None
    block_terminator = None
    block_start = 0
    tagged_lines = []
    for idx, line in enumerate(lines):
        # print(current_block_type)
        # print(line)
        normalized_line = line.lower().strip().replace(" ", "")

        # [source,...] with or without any following "----" block delimiter
        if normalized_line.startswith('[source'):
            current_block_type = 'code'
            block_start = idx
            tag = 'source_header'
        # [latex] with or without any following "++++" block delimiter
        elif normalized_line.startswith('[latex'):
            current_block_type = 'latex'
            block_start = idx
            tag = 'latex_header'
        # [note] etc with or without any following "====" block delimiter
        elif normalized_line in BLOCK_HEADERS:
            current_block_type = 'natural'
            block_start = idx
            tag = 'block_header'  # BLOCK_HEADERS[normalized_line]
        # "==", "--", '__', '++' block delimiters below block type ([note], [source], or blank line)
        elif current_block_type and idx == block_start + 1:
            if line.strip()[:2] in ('--', '==', '__', '**'):
                block_terminator = line.strip()
                tag = current_block_type + '_start'
            elif line.strip()[:2] == '++':
                block_terminator = line.strip()
                tag = current_block_type + '_start'
            # this should only happen when there's a "bare" untyped block that has started
            else:
                block_terminator = line.strip()
                tag = current_block_type
        # bare block delimiters without a block type already defined?
        elif (normalized_line in BLOCK_DELIMITERS and
                (not idx or tagged_lines[idx - 1][0] in ('blank_line', 'block_header'))):
            current_block_type = (current_block_type or BLOCK_DELIMITERS[line.rstrip()])
            block_start = idx 
            tag = current_block_type + '_start'
            block_terminator = normalized_line
        elif current_block_type and line.rstrip() == block_terminator:
            tag = current_block_type + '_end'
            current_block_type = None
            block_terminator = None
            block_start = 0
        elif current_block_type:
            tag = current_block_type
        elif not normalized_line:
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


def main(book_dir='.', include_tags=None, verbosity=1):
    print('book_dir: {}'.format(book_dir))
    print('include_tags: {}'.format(include_tags))
    print('verbosity: {}'.format(verbosity))

    sections = [(filepath, tag_lines(lines)) for filepath, lines in get_lines(book_dir)]
    if verbosity > 0:
        for filepath, tagged_lines in sections:
            if verbosity > 1:
                print('_' * 79)
                print(filepath)
                print('-' * 79)
            for tag_line in tagged_lines:
                if include_tags is None or tag_line[0] in include_tags:
                    if verbosity == 1:
                        print(tag_line[1])
                    if verbosity > 1:
                        print(tag_line)
            if verbosity > 1:
                print('=' * 79)
                print()
    return sections


if __name__ == '__main__':
    args = sys.argv[1:]
    book_dir = os.path.curdir
    if args:
        book_dir = args[0]
        args = args[1:]
    include_tags = INCLUDE_TAGS
    if args:
        try:
            verbosity = int(args[0])
            include_tags = args[1:] or include_tags
        except ValueError:
            verbosity = 1
            include_tags = args

    # print('Parsing Chapters and Appendices in: ' + book_dir)
    # print('***PRINTING LINES WITH TAGS***: ' + str(include_tags))
    main(book_dir=book_dir, include_tags=include_tags, verbosity=verbosity)
