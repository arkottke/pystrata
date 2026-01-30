Changelog
=========

All notable changes to PyStrata are documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
----------

Added
^^^^^
* Comprehensive documentation reorganization with scientific library structure
* Enhanced bibtex support with expanded reference database
* New examples gallery organization with categorized notebooks
* Improved theory section with mathematical foundations
* Developer guide with contribution guidelines

Changed
^^^^^^^
* Documentation structure follows standard scientific library layout
* Examples organized by complexity and application area
* References expanded with DOI links and complete citation information

Version 0.5.5 (2024-01-XX)
---------------------------

Added
^^^^^
* Logic tree functionality for uncertainty quantification
* Enhanced profile generation capabilities
* Miller-Rice logic tree integration
* JSON serialization for logic trees

Changed
^^^^^^^
* Build system migrated from hatch to uv + setuptools-scm
* Improved version detection with setuptools-scm integration
* Updated CI/CD workflows for uv-based builds

Fixed
^^^^^
* Dependency resolution issues with modern Python packaging
* Version detection compatibility across different installation methods

Version 0.5.4
--------------

Added
^^^^^
* Additional nonlinear soil models
* Enhanced output options
* Improved RVT calculations

Fixed
^^^^^
* Bug fixes in frequency domain calculations
* Memory optimization for large profile suites

Version 0.5.3
--------------

Added
^^^^^
* Frequency-dependent damping support
* Enhanced profile simulation capabilities
* Additional output formats

Changed
^^^^^^^
* Performance improvements for time series analysis
* Better error handling and validation

Version 0.5.2
--------------

Added
^^^^^
* RVT-based analysis capabilities
* Source theory motion generation
* Enhanced transfer function calculations

Fixed
^^^^^
* Various numerical stability improvements
* Documentation corrections

Version 0.5.1
--------------

Added
^^^^^
* Additional soil constitutive models
* Improved profile generation
* Enhanced output processing

Fixed
^^^^^
* Bug fixes in nonlinear calculations
* Memory leak fixes

Version 0.5.0
--------------

Added
^^^^^
* Major API redesign for better usability
* Modular architecture with interchangeable components
* Comprehensive test suite
* Type hints throughout codebase

Changed
^^^^^^^
* Breaking changes to API for improved consistency
* Better separation of concerns between modules
* Improved documentation structure

Removed
^^^^^^^
* Deprecated legacy interfaces
* Obsolete calculation methods

Migration Guide
---------------

**From 0.4.x to 0.5.x**

The 0.5.0 release introduced significant API changes. Key migration steps:

1. **Import Changes**

   .. code-block:: python

      # Old
      import pystrata.motion as motion

      # New
      import pystrata.motion

2. **Profile Creation**

   .. code-block:: python

      # Old
      profile = Profile(layers)

      # New
      profile = pystrata.site.Profile(layers)

3. **Calculator Interface**

   .. code-block:: python

      # Old
      calc = Calculator()
      calc.run(motion, profile)

      # New
      calc = pystrata.propagation.LinearElasticCalculator()
      calc(motion, profile, location)

**From 0.3.x to 0.4.x**

* Updated motion loading interface
* Changes to output handling
* New profile specification format

Development History
-------------------

PyStrata was originally developed as part of research activities at the University of Texas at Austin. The initial focus was on implementing and validating equivalent linear site response methods.

Key development milestones:

* **2016**: Initial development and basic time series analysis
* **2017**: Random vibration theory implementation
* **2018**: Nonlinear methods and frequency domain calculations
* **2019**: Profile generation and uncertainty analysis
* **2020**: Major API redesign and documentation improvements
* **2021**: Enhanced testing and validation
* **2022**: Performance optimizations and additional features
* **2023**: Logic tree implementation and uncertainty quantification
* **2024**: Modern packaging migration and documentation reorganization

The project continues to evolve with contributions from the earthquake engineering community and ongoing research in computational seismology.

Contributing to Changelog
--------------------------

When contributing to PyStrata:

1. Add entries to the "Unreleased" section
2. Use the standard categories: Added, Changed, Deprecated, Removed, Fixed, Security
3. Reference issue numbers where applicable
4. Follow the established format and style

For more details, see our :doc:`developer/contributing` guide.
