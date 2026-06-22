Contributing to PyStrata
========================

We welcome contributions to PyStrata! This guide will help you get started with contributing code, documentation, or examples.

Development Setup
-----------------

**Prerequisites**
    * Python 3.8 or higher
    * Git for version control
    * UV for dependency management

**Getting Started**

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/yourusername/pystrata.git
      cd pystrata

3. Set up the development environment:

   .. code-block:: bash

      uv sync
      uv run pre-commit install

4. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature-name

Development Workflow
--------------------

**Environment Management**

PyStrata uses UV for fast dependency resolution and environment management:

.. code-block:: bash

   # Install all dependencies including dev tools
   uv sync

   # Run tests
   uv run pytest

   # Run linting
   uv run ruff check src/ tests/
   uv run ruff format src/ tests/

   # Run type checking
   uv run mypy src/

**Code Quality Tools**

We use several tools to maintain code quality:

* **Ruff** - Fast Python linter and formatter
* **MyPy** - Static type checking
* **pytest** - Testing framework
* **pre-commit** - Git hooks for automated checks

**Testing**

Run the test suite to ensure your changes don't break existing functionality:

.. code-block:: bash

   # Run all tests
   uv run pytest

   # Run with coverage
   uv run pytest --cov=pystrata

   # Run specific test file
   uv run pytest tests/test_motion.py

Contributing Guidelines
-----------------------

**Code Style**

* Follow PEP 8 style guidelines
* Use descriptive variable and function names
* Add type hints to all public functions
* Maximum line length of 88 characters (Black default)

**Documentation**

* All public functions must have docstrings in NumPy format
* Include examples in docstrings where helpful
* Update relevant documentation files for new features
* Add references to scientific literature where appropriate

**Testing**

* Write tests for all new functionality
* Aim for high test coverage (>90%)
* Use descriptive test names
* Include edge cases and error conditions

**Commit Messages**

Use clear, descriptive commit messages:

.. code-block:: text

   Add support for frequency-dependent damping in RVT motions

   - Implement FDDampingMixin class
   - Add tests for frequency-dependent behavior
   - Update documentation with new damping options

   Closes #123

Types of Contributions
----------------------

**Bug Fixes**
    * Check existing issues before starting
    * Include a test that reproduces the bug
    * Provide clear description of the fix

**New Features**
    * Discuss major features in an issue first
    * Ensure backward compatibility when possible
    * Add comprehensive tests and documentation
    * Follow existing patterns and conventions

**Documentation**
    * Fix typos and improve clarity
    * Add examples and tutorials
    * Expand API documentation
    * Improve mathematical explanations

**Examples**
    * Create realistic analysis scenarios
    * Include clear explanations and context
    * Test all code in examples
    * Provide appropriate data files

Pull Request Process
--------------------

1. **Before submitting:**

   * Ensure all tests pass
   * Run the linting tools
   * Update documentation as needed
   * Add appropriate test coverage

2. **Creating the PR:**

   * Use a descriptive title
   * Reference any related issues
   * Provide clear description of changes
   * Include any breaking changes in the description

3. **Review process:**

   * Respond to reviewer feedback promptly
   * Make requested changes in new commits
   * Keep the PR focused and reasonably sized

Code Organization
-----------------

PyStrata follows a modular architecture:

**Core Modules**

* ``pystrata.motion`` - Input ground motion handling
* ``pystrata.site`` - Site profile and layer definitions
* ``pystrata.propagation`` - Wave propagation calculators
* ``pystrata.output`` - Output quantity computation
* ``pystrata.variation`` - Parameter variation and uncertainty

**Design Principles**

* **Composition over inheritance** - Use mixins and composition
* **Immutable objects** - Prefer immutable data structures
* **Clear interfaces** - Well-defined public APIs
* **Minimal dependencies** - Keep the core lightweight

Release Process
---------------

PyStrata uses semantic versioning (MAJOR.MINOR.PATCH):

* **MAJOR** - Incompatible API changes
* **MINOR** - New functionality (backward compatible)
* **PATCH** - Bug fixes (backward compatible)

Releases are automated using GitHub Actions and triggered by version tags.

Getting Help
------------

* **GitHub Issues** - Bug reports and feature requests
* **GitHub Discussions** - Questions and general discussion
* **Documentation** - Comprehensive guides and API reference

Community Guidelines
--------------------

We are committed to providing a welcoming and inclusive environment:

* Be respectful and constructive
* Focus on the technical merits
* Help newcomers get started
* Follow the code of conduct

Thank you for contributing to PyStrata!
