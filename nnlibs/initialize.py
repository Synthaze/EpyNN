#EpyNN/nnlibs/initialize.py
import nnlibs.commons.logs as clo

import pathlib
import shutil
import sys
import os


init_path = str(pathlib.Path(__file__).parent.absolute())

if not os.path.exists('./settings.py'):
    shutil.copy(init_path+'/settings.py','./settings.py')


clo.set_highlighted_excepthook()
