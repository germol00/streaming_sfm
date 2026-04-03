import os
import logging

is_debug = os.getenv('DEBUG', 'false').lower() == 'true'

if is_debug:
    LOG_LEVEL = logging.DEBUG
else:
    LOG_LEVEL = logging.NOTSET