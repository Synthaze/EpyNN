# EpyNN/epynn/initialize.py
# Local application/library specific imports
from epynn.commons.logs import set_highlighted_excepthook
from epynn.commons.library import settings_verification

# Import epynn.settings in working directory if not present
settings_verification()

# Colored excepthook
set_highlighted_excepthook()
