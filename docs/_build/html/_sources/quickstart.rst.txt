.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
 
Quick Start
==================

EpyNN requires ``python>=3.7`` and matching Python package manager ``pip3`` to install dependencies.

EpyNN is available from GitHub_ and is preferably cloned with ``git`` for command-line install.

.. _GitHub: https://github.com/Synthaze/EpyNN

Requirements
-------------

Python
~~~~~~~~

You need ``python>=3.7`` to run EpyNN.

To check your ``python3`` version on Linux, MacOS or Windows, open a command prompt and enter:

.. code-block:: bash

    python3 --version
    # e.g., Python 3.7.3

In case Python is not installed on your system or the version does not satisfy requirements, please refer to the relevant documentation available at:
`<https://wiki.python.org/moin/BeginnersGuide/Download>`_


Pip
~~~~~~~~

You need the Python package manager ``pip3`` consistent with your Python version to flawlessly install EpyNN dependencies on your system.

To check your ``pip3`` version on Linux, MacOS or Windows, open a command prompt and enter:

.. code-block:: bash

    pip3 --version
    # e.g., pip 21.1.2 from [...] (python 3.7)

In case ``pip3`` is not installed on your system or does not match your python version, please refer to the relevant documentation available at:
`<https://pip.pypa.io/en/stable/installing/>`_



Git
~~~~~~~~~~~~~

You need ``git``, a free and open source distributed version control system for command line install of EpyNN.

* Install ``git`` on Linux

*Debian based distributions*

.. code-block:: bash

    sudo apt install git

*Red-Hat based distributions*

.. code-block:: bash

    sudo yum install git


* Install ``git`` on MacOS

.. code-block:: bash

    brew install git

* Install ``git`` on Windows

The latest 64-bit version of ``git`` for Windows can be downloaded from:

`<https://git-scm.com/download/win>`_

Next, run the binary executable and follow instructions.

EpyNN Install
--------------

Linux/MacOS
~~~~~~~~~~~~~

Open a terminal and proceed with:

.. code-block:: bash

    # Use bash shell
    bash

    # Clone git repository
    git clone https://github.com/Synthaze/EpyNN

    # Change directory to EpyNN
    cd EpyNN

    # Install EpyNN dependencies
    pip3 install requirements.txt

    # Export EpyNN path in $PYTHONPATH for current session
    export PYTHONPATH=$PYTHONPATH:$PWD

Permanent export of EpyNN path in ``$PYTHONPATH``

In the same terminal session, proceed with:

* Linux

.. code-block:: bash

    # Append export instruction to the end of .bashrc file
    echo "export PYTHONPATH=$PYTHONPATH:$PWD" >> ~/.bashrc

    # Source .bashrc to refresh $PYTHONPATH
    source ~/.bashrc

* MacOS

.. code-block:: bash

    # Append export instruction to the end of .bash_profile file
    echo "export PYTHONPATH=$PYTHONPATH:$PWD" >> ~/.bash_profile

    # Source .bash_profile to refresh $PYTHONPATH
    source ~/.bash_profile



Windows
~~~~~~~~~~~~~

Open a command-prompt and proceed with:

.. code-block::

    # Clone git repository
    git clone https://github.com/Synthaze/EpyNN

    # Change directory to EpyNN
    chdir EpyNN

    # Install EpyNN dependencies
    pip3 install requirements.txt

    # Show full path of EpyNN directory
    echo %cd%

Copy the full path of EpyNN directory, then go to:
``Control Panel > System > Advanced > Environment variable``

If you already have ``PYTHONPATH`` in the ``User variables`` section, select it and click ``Edit``, otherwise click ``New`` to add it.

Paste the full path of EpyNN directory in the input field, keep in mind that paths in ``PYTHONPATH`` should be comma-separated.


Test run
-----------
