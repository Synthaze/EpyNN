# EpyNN/nnlibs/initialize.py
# Standard library imports
import pathlib
import shutil
import os

# Local application/library specific imports
from nnlibs.commons.logs import set_highlighted_excepthook
from nnlibs.commons.library import settings_verification


settings_verification()

set_highlighted_excepthook()
