import glob
import os


def get_lines(file_path):
    '''
    args:
        file_path  string  aboslute path of the Manning manuscript asciidoc files
                           i.e.: /Users/cole-home/repos/nlpinaction/manuscript

    returns:
        list of strings
    '''
    path = os.path.join(file_path, 'Chapter*')
    files = glob.glob(path)
    lines = []
    for file in files:
        with open(file, 'r') as f:
            lines += f.readlines()

    path = os.path.join(file_path, 'Appendix*')
    files = glob.glob(path)
    for file in files:
        with open(file, 'r') as f:
            lines += f.readlines()
    return lines


def tag_lines(lines):
    '''
    Mostly naively tags lines from manuscript with:
        natural
        header
        image_link
        caption
        code

    retuns:
        list of tuples  [(tag, line), ...]
    '''
    current_block_type = None
    open_block = False
    block_terminator = None
    block_start = 0
    tup_lines = []
    for idx, line in enumerate(lines):

        if line.startswith('/') or line.startswith(':'):
            continue

        if line.startswith('[source'):
            current_block_type = 'code'
            block_start = idx
            open_block = True
            continue
        if line.startswith('[TIP') or line.startswith('[NOTE'):
            current_block_type = 'text'
            block_start = idx
            open_block = True
            continue

        if open_block and idx == block_start + 1:
            if not line.startswith('--') and not line.startswith('=='):
                block_terminator = '\n'
            else:
                block_terminator = line[:2]
                continue
        if open_block and line[:2] == block_terminator:
            current_block_type = None
            open_block = False
            block_terminator = None
            block_start = 0
            continue
        if open_block and current_block_type == 'code':
            tag = 'code'
            tup_lines.append((tag, line.strip()))
            continue
        if line.startswith('='):
            tag = 'header'
            tup_lines.append((tag, line.strip()))
            continue
        if line.startswith('.'):
            tag = 'caption'
            tup_lines.append((tag, line.strip()))
            continue
        if line.startswith('image:'):
            tag = 'image_link'
            tup_lines.append((tag, line.strip()))
            continue

        if not line.strip():
            continue
        tag = 'natural'
        tup_lines.append((tag, line.strip()))
        current_block_type = None

    return tup_lines
