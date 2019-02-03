""" File utilities comparable to similarly named bash utils: rm_rf(), rm_f(), and mkdir_p() """
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import (bytes, dict, int, list, object, range, str,  # noqa
    ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
from future import standard_library
from past.builtins import basestring
standard_library.install_aliases()  # noqa

import os

from pugnlp.futil import mkdir_p  # noqa

from nlpia.constants import logging, MAX_LEN_FILEPATH
from nlpia.constants import BASE_DIR, DATA_PATH, BIGDATA_PATH, BOOK_PATH  # noqa

logger = logging.getLogger(__name__)


def ls(path, force=False):
    """ bash `ls -a`: List both file paths or directory contents (files and directories)

    >>> ls('.')
    [...]
    >>> ls('~/')
    [...]

    >>> __file__.endswith(os.path.join('nlpia', 'futil.py'))
    True
    >>> ls(__file__).endswith(os.path.join('nlpia', 'futil.py'))
    True
    """
    path = expand_filepath(path)
    logger.debug('path={}'.format(path))
    if os.path.isfile(path):
        return path
    elif os.path.isdir(path):
        return os.listdir(path)
    elif not force:
        return os.listdir(path)
    try:
        return os.listdir(path)
    except IOError:
        pass


def ls_a(path, force=False):
    """ bash `ls -a`: List both file paths or directory contents (files and directories)

    >>> path = ls(__file__)
    >>> path.endswith(os.path.join('nlpia', 'futil.py'))
    True
    """
    return ls(path, force=force)


def rm_r(path, force=False):
    """ bash `rm -r`: Recursively remove dirpath. If `force==True`, don't raise exception if path doesn't exist.

    >>> rm_r('/tmp/nlpia_dir_that_doesnt_exist_3.141591234/', force=True)
    >>> rm_r('/tmp/nlpia_dir_that_doesnt_exist_3.141591234/')
    Traceback (most recent call last):
        ...
    FileNotFoundError: [Errno 2] No such file or directory: '/tmp/nlpia_dir_that_doesnt_exist_3.141591234'
    """
    path = expand_filepath(path)
    logger.debug('path={}'.format(path))
    if os.path.isfile(path):
        return os.remove(path)
    elif os.path.isdir(path):
        try:
            return os.rmdir(path)
        except OSError:  # OSError: [Errno 66] Directory not empty: 
            pass
        except:
            if not force:
                raise
    elif not force:
        return os.rmdir(path)
    names = ls(path, force=force)
    # if ls() returns a list, path must be the full path to a directory
    if isinstance(names, list):
        if names:
            for filename in names:
                return rm_r(os.path.join(path, filename), force=force)
        else:
            os.rmdir(path)
    # if ls() returns a str, path must be the full path to a file
    elif isinstance(names, str):
        return os.remove(names, force=force)
    if force:
        return None
    return os.rmdir(path)


def rm_rf(path):
    """ bash `rm -rf`: Recursively remove dirpath. Don't raise exception if path doesn't exist.

    >>> rm_rf('/tmp/nlpia_dir_that_doesnt_exist_3.141591234/')
    """
    return rm_r(path, force=True)


def find_data_path(path):
    for fullpath in [path,
                     os.path.join(DATA_PATH, path),
                     os.path.join(BIGDATA_PATH, path),
                     os.path.join(BASE_DIR, path),
                     os.path.expanduser(os.path.join('~', path)),
                     os.path.abspath(os.path.join('.', path))
                     ]:
        if os.path.exists(fullpath):
            return fullpath
    return None


def expand_filepath(filepath):
    """ Expand any '~', '.', '*' variables in filepath.

    See also: pugnlp.futil.expand_path

    >>> len(expand_filepath('~')) > 3
    True
    """
    return os.path.abspath(os.path.expandvars(os.path.expanduser(filepath)))


def expand_filepath(filepath):
    """ Expand any '~', '.', '*' variables in filepath.

    See also: pugnlp.futil.expand_path

    >>> len(expand_filepath('~')) > 3
    True
    """
    return os.path.abspath(os.path.expandvars(os.path.expanduser(filepath)))


def ensure_open(f, mode='r'):
    r""" Return a file pointer using gzip.open if filename ends with .gz otherwise open()

    TODO: try to read a gzip rather than relying on gz extension, likewise for zip and other formats
    TODO: monkey patch the file so that .write_bytes=.write and .write writes both str and bytes

    >>> fn = os.path.join(DATA_PATH, 'pointcloud.csv.gz')
    >>> fp = ensure_open(fn)
    >>> fp
    <gzip _io.BufferedReader name='...src/nlpia/data/pointcloud.csv.gz' 0x...>
    >>> fp.closed
    False
    >>> with fp:
    ...     print(len(fp.readlines()))
    48485
    >>> fp.read()
    Traceback (most recent call last):
      ...
    ValueError: I/O operation on closed file
    >>> len(ensure_open(fp).readlines())
    48485
    >>> fn = os.path.join(DATA_PATH, 'mavis-batey-greetings.txt')
    >>> fp = ensure_open(fn)
    >>> len(fp.read())
    314
    >>> len(fp.read())
    0
    >>> len(ensure_open(fp).read())
    0
    >>> fp.close()
    >>> len(fp.read())
    Traceback (most recent call last):
      ...
    ValueError: I/O operation on closed file.
    """
    fin = f
    if isinstance(f, basestring):
        if len(f) <= MAX_LEN_FILEPATH:
            f = find_filepath(f) or f
            if f and (not hasattr(f, 'seek') or not hasattr(f, 'readlines')):
                if f.lower().endswith('.gz'):
                    return gzip.open(f, mode=mode)
                return open(f, mode=mode)
            f = fin  # reset path in case it is the text that needs to be opened with StringIO
        f = io.StringIO(f) 
    elif f.closed:
        if hasattr(f, '_write_gzip_header'):
            return gzip.open(f.name, mode=mode)
        else:
            return open(f.name, mode=mode)
    return f


def find_filepath(filename):
    """ Given a filename or path see if it exists in any of the common places datafiles might be

    >>> p = find_filepath('iq_test.csv')
    >>> p == expand_filepath(os.path.join(DATA_PATH, 'iq_test.csv'))
    True
    >>> p[-len('iq_test.csv'):]
    'iq_test.csv'
    >>> find_filepath('exponentially-crazy-filename-2.718281828459045.nonexistent')
    False
    """
    if os.path.isfile(filename):
        return filename
    for basedir in (os.path.curdir,
                    DATA_PATH,
                    BIGDATA_PATH,
                    BASE_DIR,
                    '~',
                    '~/Downloads',
                    os.path.join('/', 'tmp')):
        fullpath = expand_filepath(os.path.join(basedir, filename))
        if os.path.isfile(fullpath):
            return fullpath
    return False
