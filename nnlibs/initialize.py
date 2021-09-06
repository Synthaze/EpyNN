# EpyNN/nnlibs/initialize.py
# Local application/library specific imports
from nnlibs.commons.logs import set_highlighted_excepthook
from nnlibs.commons.library import settings_verification

# Import nnlibs.settings in working directory if not present
settings_verification()

# Colored excepthook
set_highlighted_excepthook()
