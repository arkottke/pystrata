.. _install:

Installation
============

PyStrata can be installed using pip, conda, or built from source. Python 3.10 or higher is required.

Quick Install
-------------

The easiest way to install PyStrata is using pip:

.. code-block:: bash

   pip install pystrata

Or using conda:

.. code-block:: bash

   conda install -c conda-forge pystrata

Dependencies
------------

PyStrata requires the following packages:

* **numpy** -- Fast numerical arrays and linear algebra
* **scipy** -- Scientific computing and signal processing
* **matplotlib** -- Plotting and visualization
* **xarray** -- Labeled multi-dimensional arrays
* **numba** -- Just-in-time compilation for performance
* **pyrvt** -- Random vibration theory calculations
* **pykooh** -- Peak factor calculations

These dependencies are automatically installed when using pip or conda.

Development Installation
------------------------

To install PyStrata for development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/arkottke/pystrata.git
   cd pystrata
   pip install -e .

For development with all optional dependencies:

.. code-block:: bash

   git clone https://github.com/arkottke/pystrata.git
   cd pystrata
   pip install -e ".[test,docs]"

Using UV (Recommended for Development)
--------------------------------------

For the fastest development setup, use UV:

.. code-block:: bash

   git clone https://github.com/arkottke/pystrata.git
   cd pystrata
   uv sync
   uv run pytest  # Run tests

System Requirements
-------------------

* **Python**: 3.10 or higher
* **Operating System**: Windows, macOS, or Linux
* **Memory**: 4GB RAM minimum, 8GB recommended for large analyses
* **Storage**: 1GB free space for installation and examples

Optional Dependencies
---------------------

Additional packages for enhanced functionality:

* **jupyter** -- For running example notebooks
* **pandas** -- Enhanced data handling
* **openpyxl** -- Excel file support

Install optional dependencies:

.. code-block:: bash

   pip install pystrata[test,docs]

Verification
------------

Test your installation:

.. code-block:: python

   import pystrata
   print(pystrata.__version__)

   # Run a simple example
   motion = pystrata.motion.SourceTheoryRvtMotion(6.5, 20, "wna")
   print("Installation successful!")

Troubleshooting
---------------

**Import Errors**
   If you encounter import errors, ensure all dependencies are installed:

   .. code-block:: bash

      pip install --upgrade numpy scipy matplotlib

**Permission Errors**
   On some systems, you may need to use ``--user`` flag:

   .. code-block:: bash

      pip install --user pystrata

**Environment Conflicts**
   Consider using a virtual environment:

   .. code-block:: bash

      python -m venv pystrata-env
      source pystrata-env/bin/activate  # On Windows: pystrata-env\Scripts\activate
      pip install pystrata

For additional help, see the GitHub repository issues page or the documentation.

.. _Windows 32-bit: http://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
.. _Windows 64-bit: http://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
.. _OS-X: http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

After the installer is finished, install the required dependencies by opening a
terminal. On Windows, this is best accomplished with ``Windows Key + r``, enter
``cmd``. Next enter the following command::

  conda install --yes setuptools numpy scipy matplotlib pytest

On Windows, the text can copied and pasted if "Quick Edit" mode is enabled. To
enable this feature, right click on the icon in the upper left portion of the
window, and select "Properties", and then check the "Quick Edit Mode" check box
within the "Edit Options" group. Copy the text, and then paste it by click the
right mouse button.

Now that the dependencies have been installed, install or upgrade pyRVT and
pystrata using pip::

  pip install --upgrade pyrvt pystrata

You should now have pystrata completely installed. Next, read about
:ref:`using <usage>` pystrata.
