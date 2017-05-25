import logging
import sys

import nlpia.constants  # noqa  this will setup logging to go to stdout or loggly (if you ask for it)


def stdout_logging(loglevel=logging.INFO):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(lineno)d: %(message)s"

    logging.config.dictConfig(level=loglevel, stream=sys.stdout,
                              format=logformat, datefmt="%Y-%m-%d %H:%M:%S")
