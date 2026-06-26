
.. include:: include/links.rst

Installation
============

Clone the repo
--------------

To download the DAP software and associated data, clone the `mangadap
GitHub repo`_ by executing:

    .. code-block:: console

        git clone https://github.com/briandigiorgio/NIRVANA.git

This will create a ``NIRVANA`` directory in the current directory.
Although we try to keep the ``master`` branch of the repository stable,
we recommend using the most recent tag.  You can do so by executing:

    .. code-block:: console

        cd NIRVANA
        ./checkout_current_tag

Install Python 3
----------------

NIRVANA is supported for Python 3 only. To install Python, you can do
so along with a full package manager, like `Anaconda`_, or you can
install python 3 directly from `python.org`_.


Install the NIRVANA from source
-------------------------------

The preferred method to install NIRVANA and ensure its dependencies are
met is to, from the top-level, ``NIRVANA`` directory, run:

.. code-block:: console

    pip install -e .

This approach is preferred because it eases uninstalling the code:

.. code-block:: console
    
    pip uninstall nirvana

Installation in this way should also mean that changes made to the code
should take immediate effect when you restart the calling python
session.

----

To install only the NIRVANA dependencies, run:

.. code-block:: console

    pip install -r requirements.txt


Test your installation
----------------------

To test the installation, use ``pytest``:

.. code-block:: console

    cd nirvana/tests
    pytest . -W ignore

Some tests requires a set of "remote" data that are not located in
the repo for space considerations. Downloading the data used by these
tests currently requires `SDSS Collaboration Access`_. The link in
the last sentence points to a description of how this access is
granted for Marvin using a ``~\.netrc`` file.  NIRVANA uses the same
``~\.netrc`` file to authenticate access to the ``data.sdss.org``
host for downloading the test data. Once you have your ``~\.netrc``
file, you can download the necessary test data and rerun the tests to
include usage of that data like this:

    .. code-block:: console

        python3 download_test_data.py
        cd nirvana/tests
        pytest . -W ignore

Problems?
---------

We have limited support to offer installation help. However, if you
have problems, particularly those that you think may be a more
general problem, please `Submit an issue`_.

