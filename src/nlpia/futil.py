""" File utilities comparable to similarly named bash utils: rm_rf(), rm_f(), and mkdir_p() """
import os
from nlpia.constants import logging
from pugnlp.futil import expand_path, mkdir_p  # noqa

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
    path = expand_path(path)
    logger.debug('path={}'.format(path))
    if os.path.isfile(path):
        return path
    elif os.path.isdir(path):
        return os.listdir(path)
    elif not force:
        return os.listdir(path)
    try:
        return os.listdir(path)
    except:
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
    path = expand_path(path)
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

